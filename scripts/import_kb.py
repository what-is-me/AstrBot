#!/usr/bin/env python3
"""将离线构建的知识库导入到 AstrBot 服务器

极简实现：只依赖标准库 + sqlite3，通过原生 SQL 把元数据写入 kb.db，
并复制 doc.db / index.faiss 到目标目录。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将离线构建的知识库导入 AstrBot 服务器",
    )
    parser.add_argument("--package-dir", required=True, help="离线包目录")
    parser.add_argument(
        "--embedding-provider-id",
        required=True,
        help="服务器端要注册的 embedding provider id",
    )
    parser.add_argument(
        "--astrbot-root",
        default=os.environ.get("ASTRBOT_ROOT", ""),
        help="AstrBot 根目录（默认 ASTRBOT_ROOT 环境变量或当前目录）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的知识库目录和元数据",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()

    package_dir = Path(args.package_dir)
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found in {package_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kb_id = manifest["kb_id"]
    kb_name = manifest["kb_name"]

    astrbot_root = Path(args.astrbot_root) if args.astrbot_root else Path.cwd()
    kb_root = astrbot_root / "data" / "knowledge_base"
    kb_db_path = kb_root / "kb.db"
    target_dir = kb_root / kb_id

    if not kb_db_path.exists():
        raise SystemExit(f"kb.db not found: {kb_db_path}")

    # 复制索引文件
    if target_dir.exists():
        if not args.force:
            raise SystemExit(
                f"Target directory already exists: {target_dir}. Use --force to overwrite.",
            )
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(package_dir / "doc.db", target_dir / "doc.db")
    shutil.copy2(package_dir / "index.faiss", target_dir / "index.faiss")

    # 写入 kb.db
    conn = sqlite3.connect(str(kb_db_path))
    cur = conn.cursor()

    # 检查冲突
    cur.execute("SELECT 1 FROM knowledge_bases WHERE kb_id = ?", (kb_id,))
    if cur.fetchone():
        if not args.force:
            raise SystemExit(
                f"kb_id {kb_id} already exists in kb.db. Use --force to overwrite.",
            )
        cur.execute("DELETE FROM knowledge_bases WHERE kb_id = ?", (kb_id,))
        cur.execute("DELETE FROM kb_documents WHERE kb_id = ?", (kb_id,))

    cur.execute("SELECT 1 FROM knowledge_bases WHERE kb_name = ?", (kb_name,))
    if cur.fetchone():
        raise SystemExit(f"kb_name {kb_name} already exists in kb.db.")

    now = now_iso()
    cur.execute(
        """
        INSERT INTO knowledge_bases (
            kb_id, kb_name, description, emoji,
            embedding_provider_id, rerank_provider_id,
            chunk_size, chunk_overlap,
            top_k_dense, top_k_sparse, top_m_final,
            created_at, updated_at, doc_count, chunk_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            kb_id,
            kb_name,
            f"Imported from {package_dir.name}",
            "📚",
            args.embedding_provider_id,
            None,
            512,
            50,
            50,
            50,
            5,
            now,
            now,
            manifest.get("doc_count", 0),
            manifest.get("chunk_count", 0),
        ),
    )

    for doc in manifest.get("documents", []):
        cur.execute(
            """
            INSERT INTO kb_documents (
                doc_id, kb_id, doc_name, file_type, file_size,
                file_path, chunk_count, media_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc["doc_id"],
                kb_id,
                doc["doc_name"],
                doc.get("file_type", "directory"),
                doc.get("file_size", 0),
                "",
                doc.get("chunk_count", 0),
                0,
                now,
                now,
            ),
        )

    conn.commit()
    conn.close()

    print(f"Imported KB {kb_id} ({kb_name})")
    print(f"  docs: {manifest.get('doc_count', 0)}")
    print(f"  chunks: {manifest.get('chunk_count', 0)}")
    print(f"  target: {target_dir}")
    print("Please restart AstrBot to load the new knowledge base.")


if __name__ == "__main__":
    main()
