"""离线构建知识库的公共工具

被 scripts/build_kb_offline.py 和 scripts/import_kb.py 复用。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from astrbot.core.db.vec_db.faiss_impl.document_storage import DocumentStorage
    from astrbot.core.provider.provider import EmbeddingProvider


def get_tqdm():
    """返回 tqdm 进度条类，未安装时返回 None。"""
    try:
        from tqdm.asyncio import tqdm

        return tqdm
    except Exception:
        return None


def tqdm_progress_bar(iterable, desc, total=None, disable=False):
    """创建进度条，未安装 tqdm 时回退为无进度条迭代。"""
    tqdm_cls = get_tqdm()
    if tqdm_cls is None:
        if desc:
            print(f"[start] {desc}")
        yield from iterable
        if desc:
            print(f"[done] {desc}")
        return
    yield from tqdm_cls(iterable, desc=desc, total=total, disable=disable)


@dataclass
class BuildState:
    """构建状态，用于断点续跑。"""

    kb_id: str
    kb_name: str
    completed_doc_names: list[str] = field(default_factory=list)
    in_progress_doc_name: str | None = None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> BuildState:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


@dataclass
class Manifest:
    """离线包清单，服务端导入时读取。"""

    kb_id: str
    kb_name: str
    embedding_provider_id: str
    embedding_dim: int
    chunk_count: int
    doc_count: int
    documents: list[dict] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


@dataclass
class DocChunks:
    """一个目录文档及其对应的 chunks。"""

    doc_id: str
    doc_name: str
    file_type: str
    file_size: int
    chunks: list[str] = field(default_factory=list)


class DirectoryLoader:
    """按目录聚合文档的加载器。

    ``base_dir`` 下的每个一级子目录被视为一个文档；子目录内的每个文件
    默认作为一个整块（chunk）。``.txt`` / ``.md`` 使用 ``TextParser``，
    ``.json`` 使用 ``JsonParser``。

    当文件内容长度超过 ``max_chunk_length`` 时，会使用递归字符分割器
    将其拆分为多个 chunk，避免超出 embedding 模型的 token 上限。
    """

    def __init__(
        self,
        base_dir: str | Path,
        max_chunk_length: int | None = None,
        chunk_overlap: int = 100,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.max_chunk_length = max_chunk_length
        self.chunk_overlap = chunk_overlap
        self._chunker = None
        if max_chunk_length and max_chunk_length > 0:
            from astrbot.core.knowledge_base.chunking.recursive import (
                RecursiveCharacterChunker,
            )

            self._chunker = RecursiveCharacterChunker(
                chunk_size=max_chunk_length,
                chunk_overlap=chunk_overlap,
            )

    async def load(self) -> list[DocChunks]:
        """加载所有目录文档。"""
        from astrbot.core.knowledge_base.parsers.json_parser import JsonParser
        from astrbot.core.knowledge_base.parsers.text_parser import TextParser

        text_parser = TextParser()
        json_parser = JsonParser()

        docs: list[DocChunks] = []
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory does not exist: {self.base_dir}")

        entries = sorted(
            [p for p in self.base_dir.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )
        if not entries:
            # 如果根目录下没有子目录，把整个 base_dir 作为一个文档
            entries = [self.base_dir]

        for doc_dir in entries:
            chunks: list[str] = []
            file_size = 0
            files = sorted(
                [p for p in doc_dir.rglob("*") if p.is_file()],
                key=lambda p: str(p.relative_to(doc_dir)),
            )
            for file_path in files:
                if file_path.name.startswith("."):
                    continue
                ext = file_path.suffix.lower()
                if ext not in {".txt", ".md", ".markdown", ".json"}:
                    print(f"  [skip] unsupported extension {ext}: {file_path}")
                    continue
                content = await asyncio.to_thread(file_path.read_bytes)
                file_size += len(content)
                try:
                    if ext == ".json":
                        result = await json_parser.parse(content, file_path.name)
                    else:
                        result = await text_parser.parse(content, file_path.name)
                except Exception as e:
                    print(f"  [warn] failed to parse {file_path}: {e}")
                    continue
                text = result.text.strip()
                if not text:
                    continue
                if self._chunker and len(text) > self.max_chunk_length:
                    print(
                        f"  [split] {file_path.relative_to(doc_dir)} "
                        f"({len(text)} chars) into chunks <= {self.max_chunk_length}",
                    )
                    split_chunks = await self._chunker.chunk(text)
                    header = f"[文档: {doc_dir.name} | 文件: {file_path.name}]\n"
                    chunks.extend(header + chunk for chunk in split_chunks)
                else:
                    chunks.append(text)

            doc_name = doc_dir.name if doc_dir != self.base_dir else self.base_dir.name
            docs.append(
                DocChunks(
                    doc_id=str(uuid.uuid4()),
                    doc_name=doc_name,
                    file_type="directory",
                    file_size=file_size,
                    chunks=chunks,
                ),
            )

        return [d for d in docs if d.chunks]


def load_embedding_provider(provider_config: dict) -> EmbeddingProvider:
    """根据 provider_config 中的 type 实例化 Embedding Provider。

    通过导入 ``astrbot.core.provider.sources.<type>_source`` 注册 provider，
    然后从 ``provider_cls_map`` 获取对应类。
    """
    from astrbot.core.provider.register import provider_cls_map

    provider_type = provider_config.get("type")
    if not provider_type:
        raise ValueError("provider_config 必须包含 'type' 字段")

    meta = provider_cls_map.get(provider_type)
    if not meta:
        module_name = f"astrbot.core.provider.sources.{provider_type}_source"
        try:
            __import__(module_name)
        except ModuleNotFoundError as e:
            raise ValueError(
                f"无法找到类型为 {provider_type} 的 provider 模块: {module_name}",
            ) from e

        meta = provider_cls_map.get(provider_type)
        if not meta:
            raise ValueError(
                f"Provider type {provider_type} 未注册。已注册的类型: {list(provider_cls_map.keys())}",
            )

    cls = meta.cls_type
    provider = cls(provider_config, {})
    return provider


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def reset_output_dir(path: str | Path) -> Path:
    """删除并重建输出目录。"""
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def choose_ivfpq_params(
    dimension: int,
    n_vectors: int,
    nlist: int | None = None,
    m: int | None = None,
    nbits: int | None = None,
) -> dict | None:
    """选择 IVFPQ 参数，若不适合则返回 None（应回退 FlatL2）。"""

    if nlist is None:
        nlist = max(16, int(4 * math.sqrt(n_vectors)))
    if nbits is None:
        nbits = 8

    if n_vectors < nlist * 30 or n_vectors < 1000:
        return None

    if m is None:
        for candidate in (dimension // 8, dimension // 4, dimension // 2, dimension):
            if candidate > 0 and dimension % candidate == 0:
                m = candidate
                break
    if m is None or dimension % m != 0:
        return None

    return {"nlist": nlist, "m": m, "nbits": nbits}


def create_or_load_flat_index(index_path: str | Path, dimension: int):
    """创建或加载一个 FlatL2 + IndexIDMap2 索引。"""
    import faiss

    index_path = Path(index_path)
    if index_path.exists():
        return faiss.read_index(str(index_path))
    base = faiss.IndexFlatL2(dimension)
    return faiss.IndexIDMap2(base)


def save_index(index, index_path: str | Path) -> None:
    """保存 FAISS 索引。"""
    import faiss

    faiss.write_index(index, str(index_path))


def convert_flat_to_ivfpq(
    flat_index_path: str | Path,
    dimension: int,
    params: dict,
    output_path: str | Path,
) -> None:
    """将已构建的 FlatL2 索引转换为 IVFPQ 索引。"""
    import faiss

    flat_index = faiss.read_index(str(flat_index_path))
    ntotal = flat_index.ntotal
    if ntotal == 0:
        raise ValueError("Cannot convert empty index to IVFPQ")

    ids = np.array(
        [flat_index.id_map.at(i) for i in range(ntotal)],
        dtype=np.int64,
    )
    vectors = np.array(
        [flat_index.reconstruct(int(i)) for i in ids],
        dtype=np.float32,
    )

    quantizer = faiss.IndexFlatL2(dimension)
    ivfpq = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        params["nlist"],
        params["m"],
        params["nbits"],
    )
    index = faiss.IndexIDMap2(ivfpq)

    train_count = min(ntotal, max(params["nlist"] * 100, 65536))
    if train_count < ntotal:
        rng = np.random.default_rng(42)
        train_vectors = vectors[rng.choice(ntotal, train_count, replace=False)]
    else:
        train_vectors = vectors

    if not index.is_trained:
        index.train(train_vectors)

    index.add_with_ids(vectors, ids)
    faiss.write_index(index, str(output_path))


async def clean_doc_chunks(
    doc_storage: DocumentStorage,
    faiss_index,
    doc_id: str,
) -> None:
    """清理某个文档在 doc.db 和 FAISS 索引中的残留 chunks。"""
    docs = await doc_storage.get_documents(
        metadata_filters={"kb_doc_id": doc_id},
        offset=None,
        limit=None,
    )
    if not docs:
        return

    ids_to_remove = [int(doc["id"]) for doc in docs]
    await doc_storage.delete_documents(metadata_filters={"kb_doc_id": doc_id})
    if faiss_index is not None and ids_to_remove:
        faiss_index.remove_ids(np.array(ids_to_remove, dtype=np.int64))


def format_elapsed(seconds: float) -> str:
    """将秒数格式化为易读字符串。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def dump_json(obj, path: str | Path) -> None:
    """将对象以 JSON 形式写入文件。"""
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
