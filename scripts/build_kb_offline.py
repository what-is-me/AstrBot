#!/usr/bin/env python3
"""离线构建 AstrBot 知识库

在本地完成 embedding、FAISS 索引构建，生成可直接上传到服务器的知识库包。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
from kb_offline_common import (
    BuildState,
    DirectoryLoader,
    Manifest,
    choose_ivfpq_params,
    clean_doc_chunks,
    convert_flat_to_ivfpq,
    create_or_load_flat_index,
    ensure_dir,
    format_elapsed,
    load_embedding_provider,
    reset_output_dir,
    save_index,
    tqdm_progress_bar,
)

from astrbot.core import logger
from astrbot.core.db.vec_db.faiss_impl.document_storage import DocumentStorage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在本地离线构建 AstrBot 知识库",
    )
    parser.add_argument("--base-dir", required=True, help="源数据根目录")
    parser.add_argument("--kb-name", required=True, help="知识库名称")
    parser.add_argument(
        "--kb-id",
        default="",
        help="知识库 ID（默认自动生成 UUID）",
    )
    parser.add_argument(
        "--output-dir",
        default="./kb_export",
        help="输出目录（默认 ./kb_export）",
    )
    parser.add_argument(
        "--provider-config",
        required=True,
        help="Embedding Provider 配置 JSON 文件路径",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="embedding 批大小",
    )
    parser.add_argument(
        "--tasks-limit",
        type=int,
        default=3,
        help="embedding 并发任务数",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="embedding 失败重试次数",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="每个 embedding 请求之间的间隔秒数（用于限流，默认 0）",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="日志级别（默认 INFO）",
    )
    parser.add_argument(
        "--max-chunk-length",
        type=int,
        default=8100,
        help="单个 chunk 最大字符数，超过则拆分（默认 8100）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="拆分 chunk 时的重叠字符数（默认 200）",
    )
    parser.add_argument(
        "--index-type",
        choices=["ivfpq", "flat"],
        default="ivfpq",
        help="FAISS 索引类型（默认 ivfpq）",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=None,
        help="IVFPQ nlist（默认自动计算）",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="IVFPQ m（默认自动计算）",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=None,
        help="IVFPQ nbits（默认 8）",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="全部文档完成后用全局向量重新训练 IVFPQ",
    )
    parser.add_argument(
        "--single-embedding",
        action="store_true",
        help="逐个 chunk 请求 embedding（兼容不支持 batch 的 provider）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新构建（删除已有输出目录）",
    )
    return parser.parse_args()


def find_existing_kb_id(output_dir: Path) -> str | None:
    """在输出目录中查找已有知识包，返回其 kb_id。"""
    if not output_dir.exists():
        return None
    for item in output_dir.iterdir():
        if not item.is_dir():
            continue
        state_file = item / "build_state.json"
        manifest_file = item / "manifest.json"
        if state_file.exists():
            data = json.loads(state_file.read_text(encoding="utf-8"))
            if data.get("kb_id"):
                return data["kb_id"]
        if manifest_file.exists():
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if data.get("kb_id"):
                return data["kb_id"]
    return None


def _preview_text(text: str, max_len: int = 200) -> str:
    """返回文本预览，用于错误诊断。"""
    if len(text) <= max_len:
        return text.replace("\n", " ")
    return (text[: max_len // 2] + " ... " + text[-max_len // 2 :]).replace("\n", " ")


async def get_embeddings_single(
    provider,
    texts: list[str],
    max_retries: int = 3,
    request_interval: float = 0.0,
    progress_callback=None,
) -> list[list[float]]:
    """逐个文本获取 embedding，兼容不支持 batch 的 provider。"""
    vectors: list[list[float]] = []
    total = len(texts)
    for idx, text in enumerate(texts):
        for attempt in range(max_retries):
            try:
                vector = await provider.get_embedding(text)
                vectors.append(vector)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        f"  [diag] failing chunk {idx}: len={len(text)}, "
                        f"preview={_preview_text(text)!r}",
                    )
                    raise Exception(
                        f"chunk {idx} 请求失败，已重试 {max_retries} 次: {e!s}",
                    ) from e
                await asyncio.sleep(2**attempt)
        if progress_callback:
            await progress_callback(idx + 1, total)
        if request_interval > 0 and idx < total - 1:
            await asyncio.sleep(request_interval)
    return vectors


async def get_embeddings_batch_with_interval(
    provider,
    texts: list[str],
    batch_size: int = 16,
    max_retries: int = 16,
    request_interval: float = 0.0,
    progress_callback=None,
) -> list[list[float]]:
    """批量获取 embedding，支持请求间隔以应对限流。"""
    vectors: list[list[float]] = []
    total = len(texts)
    batch_count = (total + batch_size - 1) // batch_size
    for batch_idx in range(batch_count):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = texts[start:end]
        for attempt in range(max_retries):
            try:
                batch_vectors = await provider.get_embeddings(batch)
                vectors.extend(batch_vectors)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(
                        f"批次 {batch_idx} 处理失败，已重试 {max_retries} 次: {e!s}",
                    ) from e
                await asyncio.sleep(2**attempt)
        if progress_callback:
            await progress_callback(end, total)
        if request_interval > 0 and batch_idx < batch_count - 1:
            await asyncio.sleep(request_interval)
    return vectors


async def main() -> None:
    args = parse_args()
    level = getattr(logging, args.log_level)
    logger.setLevel(level)
    logging.getLogger().setLevel(level)
    start_time = time.time()

    # 输出目录与知识库 ID
    output_dir = Path(args.output_dir)
    kb_id = args.kb_id or find_existing_kb_id(output_dir) or str(uuid.uuid4())
    package_dir = output_dir / kb_id

    if args.force and package_dir.exists():
        reset_output_dir(package_dir)
    ensure_dir(package_dir)

    # 加载 provider 配置
    provider_config = json.loads(Path(args.provider_config).read_text(encoding="utf-8"))
    ep = load_embedding_provider(provider_config)

    # 确定维度
    embedding_dim = ep.get_dim()
    if embedding_dim <= 0:
        sample_vec = await ep.get_embedding("astrbot")
        embedding_dim = len(sample_vec)
    print(f"Embedding dimension: {embedding_dim}")

    # 加载目录文档
    print(f"Loading documents from {args.base_dir} ...")
    loader = DirectoryLoader(
        args.base_dir,
        max_chunk_length=args.max_chunk_length,
        chunk_overlap=args.chunk_overlap,
    )
    docs = await loader.load()
    if not docs:
        raise SystemExit("No documents found in base directory.")
    total_chunks = sum(len(d.chunks) for d in docs)
    print(f"Found {len(docs)} documents, {total_chunks} chunks")

    # 初始化 doc.db
    doc_db_path = package_dir / "doc.db"
    doc_storage = DocumentStorage(str(doc_db_path))
    await doc_storage.initialize()

    # 初始化或加载 FAISS 索引（先用 FlatL2 支持增量写入）
    index_path = package_dir / "index.faiss"
    flat_index = create_or_load_flat_index(index_path, embedding_dim)

    # 状态与清单
    state_path = package_dir / "build_state.json"
    manifest_path = package_dir / "manifest.json"

    state = BuildState(kb_id=kb_id, kb_name=args.kb_name)
    if state_path.exists():
        state = BuildState.load(state_path)
        print(
            f"Resuming from previous build, {len(state.completed_doc_names)} docs done"
        )

    manifest = Manifest(
        kb_id=kb_id,
        kb_name=args.kb_name,
        embedding_provider_id=provider_config.get("id", ""),
        embedding_dim=embedding_dim,
        chunk_count=0,
        doc_count=0,
    )
    if manifest_path.exists():
        manifest = Manifest.load(manifest_path)

    # 断点续跑时，用 state 重建 manifest，避免上次中断留下不一致数据
    completed_set = set(state.completed_doc_names)
    manifest.documents = [
        d for d in manifest.documents if d.get("doc_name") in completed_set
    ]
    manifest.doc_count = len(manifest.documents)
    manifest.chunk_count = sum(d.get("chunk_count", 0) for d in manifest.documents)
    manifest.save(manifest_path)

    # 处理每个文档
    doc_iter = tqdm_progress_bar(docs, desc="Building docs", total=len(docs))
    for doc in doc_iter:
        if doc.doc_name in state.completed_doc_names:
            print(f"[skip] {doc.doc_name} (already done)")
            continue

        # 如果上次中断时正在处理这个文档，先清理残留
        if state.in_progress_doc_name == doc.doc_name:
            print(f"[resume] cleaning partial data for {doc.doc_name}")
            await clean_doc_chunks(doc_storage, flat_index, doc.doc_id)

        state.in_progress_doc_name = doc.doc_name
        state.save(state_path)

        print(f"[embed] {doc.doc_name}: {len(doc.chunks)} chunks")

        # embedding 进度回调
        def make_progress_cb(desc):
            async def _cb(current, total):
                print(f"\r  {desc}: {current}/{total}", end="", flush=True)

            return _cb

        try:
            if args.single_embedding:
                vectors = await get_embeddings_single(
                    ep,
                    doc.chunks,
                    max_retries=args.max_retries,
                    request_interval=args.request_interval,
                    progress_callback=make_progress_cb("embedding"),
                )
            elif args.request_interval > 0:
                # 使用自定义批量循环，支持请求间隔
                vectors = await get_embeddings_batch_with_interval(
                    ep,
                    doc.chunks,
                    batch_size=args.batch_size,
                    max_retries=args.max_retries,
                    request_interval=args.request_interval,
                    progress_callback=make_progress_cb("embedding"),
                )
            else:
                vectors = await ep.get_embeddings_batch(
                    doc.chunks,
                    batch_size=args.batch_size,
                    tasks_limit=args.tasks_limit,
                    max_retries=args.max_retries,
                    progress_callback=make_progress_cb("embedding"),
                )
        finally:
            print()  # newline after progress

        if len(vectors) != len(doc.chunks):
            raise RuntimeError(
                f"Embedding count mismatch for {doc.doc_name}: "
                f"expected {len(doc.chunks)}, got {len(vectors)}",
            )

        # 生成 chunk 级 doc_id 与 metadata
        chunk_doc_ids = [str(uuid.uuid4()) for _ in doc.chunks]
        metadatas = [
            {
                "kb_id": kb_id,
                "kb_doc_id": doc.doc_id,
                "chunk_index": idx,
            }
            for idx in range(len(doc.chunks))
        ]

        # 写入 doc.db
        int_ids = await doc_storage.insert_documents_batch(
            chunk_doc_ids,
            doc.chunks,
            metadatas,
        )

        # 写入 FAISS（FlatL2）
        vectors_array = np.asarray(vectors, dtype=np.float32)
        if vectors_array.shape[1] != embedding_dim:
            raise RuntimeError(
                f"Vector dimension mismatch for {doc.doc_name}: "
                f"expected {embedding_dim}, got {vectors_array.shape[1]}",
            )
        flat_index.add_with_ids(
            vectors_array,
            np.array(int_ids, dtype=np.int64),
        )
        save_index(flat_index, index_path)

        # 更新清单
        manifest.doc_count += 1
        manifest.chunk_count += len(doc.chunks)
        manifest.documents.append(
            {
                "doc_id": doc.doc_id,
                "doc_name": doc.doc_name,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "chunk_count": len(doc.chunks),
            },
        )
        manifest.save(manifest_path)

        state.in_progress_doc_name = None
        state.completed_doc_names.append(doc.doc_name)
        state.save(state_path)

    await doc_storage.close()

    # 最终转换 IVFPQ
    if args.index_type == "ivfpq":
        ntotal = flat_index.ntotal
        params = choose_ivfpq_params(
            embedding_dim,
            ntotal,
            args.nlist,
            args.m,
            args.nbits,
        )
        if params:
            print(
                f"Converting {ntotal} vectors to IVFPQ "
                f"(nlist={params['nlist']}, m={params['m']}, nbits={params['nbits']}) ...",
            )
            convert_flat_to_ivfpq(index_path, embedding_dim, params, index_path)
            print("IVFPQ conversion done")
        else:
            print(
                f"Too few vectors ({ntotal}) for IVFPQ, keeping FlatL2 index.",
            )
    elif args.retrain:
        print("--retrain is only effective with --index-type ivfpq")

    # 关闭 provider
    if hasattr(ep, "terminate"):
        await ep.terminate()

    elapsed = time.time() - start_time
    print(
        f"Build complete: {manifest.doc_count} docs, {manifest.chunk_count} chunks "
        f"in {format_elapsed(elapsed)}",
    )
    print(f"Output: {package_dir}")


if __name__ == "__main__":
    asyncio.run(main())
