"""
知识库服务：文档切分 + Qdrant 向量存储 + 检索
"""

import logging
import uuid
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

from app.config import config

logger = logging.getLogger(__name__)

# ── 文本切分 ──────────────────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    递归字符切分：优先按段落 > 换行 > 句号 > 逗号切，
    保证每块不超过 chunk_size 字符，相邻块有 overlap 字符重叠。
    """
    separators = ["\n\n", "\n", "。", "；", "，", " ", ""]
    return _split_recursive(text.strip(), chunk_size, overlap, separators)


def _split_recursive(text: str, chunk_size: int, overlap: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = ""
    for s in separators:
        if s == "" or s in text:
            sep = s
            break

    if sep == "":
        # 强制按字符数切
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    parts = text.split(sep)
    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                # 递归用更细粒度的分隔符
                sub = _split_recursive(part, chunk_size, overlap, separators[separators.index(sep)+1:])
                chunks.extend(sub)
                current = sub[-1][-overlap:] if sub and overlap else ""
            else:
                current = part.strip()

    if current:
        chunks.append(current)

    # 加 overlap：把上一块末尾拼到下一块开头
    if overlap > 0 and len(chunks) > 1:
        merged = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = merged[-1][-overlap:]
            merged.append(tail + chunks[i])
        return merged

    return chunks


# ── 文档解析 ──────────────────────────────────────────────────────────────────

def extract_text(file_path: str) -> str:
    """从文件提取纯文本，支持 txt / pdf / docx"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix in (".docx", ".doc"):
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"不支持的文件类型: {suffix}，仅支持 txt / pdf / docx")


# ── Embedding ─────────────────────────────────────────────────────────────────

class EmbeddingClient:
    """调用 SiliconFlow embedding 接口"""

    def __init__(self):
        ai_cfg = config.ai
        self._client = OpenAI(
            api_key=ai_cfg.get("api_key"),
            base_url=ai_cfg.get("base_url"),
        )
        self._model = config.embedding.get("model", "BAAI/bge-m3")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量 embedding，每次最多 32 条"""
        results = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            results.extend([item.embedding for item in resp.data])
        return results

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


# ── Qdrant 知识库 ─────────────────────────────────────────────────────────────

class KnowledgeService:

    def __init__(self):
        qdrant_cfg = config.qdrant
        self._client = QdrantClient(
            host=qdrant_cfg.get("host", "localhost"),
            port=qdrant_cfg.get("port", 6333),
            check_compatibility=False,
        )
        self._collection = qdrant_cfg.get("collection", "knowledge")
        self._dim = config.embedding.get("dimension", 1024)
        self._embedder = EmbeddingClient()
        self._ensure_collection()

    def _ensure_collection(self):
        """集合不存在时自动创建"""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            logger.info(f"✅ Qdrant 集合已创建: {self._collection}")

    # ── 写入 ──────────────────────────────────────────────────────────────────

    def ingest_text(self, text: str, doc_id: str, category_id: str,
                    source: str = "", chunk_size: int = 500, overlap: int = 50) -> int:
        """切分纯文本并写入 Qdrant，返回写入的 chunk 数量"""
        chunks = split_text(text, chunk_size, overlap)
        if not chunks:
            return 0

        vectors = self._embedder.embed(chunks)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "doc_id": doc_id,
                    "category_id": category_id,
                    "source": source,
                    "chunk_index": i,
                    "content": chunk,
                    "create_time": int(time.time()),
                },
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        logger.info(f"✅ 写入 {len(points)} 个 chunk，doc_id={doc_id}")
        return len(points)

    def ingest_file(self, file_path: str, doc_id: str, category_id: str,
                    chunk_size: int = 500, overlap: int = 50) -> int:
        """解析文件并写入 Qdrant"""
        text = extract_text(file_path)
        source = Path(file_path).name
        return self.ingest_text(text, doc_id, category_id, source, chunk_size, overlap)

    # ── 删除 ──────────────────────────────────────────────────────────────────

    def delete_document(self, doc_id: str):
        """删除某文档的所有 chunk"""
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        logger.info(f"🗑 已删除 doc_id={doc_id} 的所有 chunk")

    # ── 检索 ──────────────────────────────────────────────────────────────────

    def search(self, query: str, category_id: Optional[str] = None,
               top_k: int = 5) -> List[dict]:
        """向量检索，可按 category_id 过滤"""
        vec = self._embedder.embed_one(query)
        query_filter = None
        if category_id:
            query_filter = Filter(
                must=[FieldCondition(key="category_id", match=MatchValue(value=category_id))]
            )
        hits = self._client.query_points(
            collection_name=self._collection,
            query=vec,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {
                "score": round(h.score, 4),
                "content": h.payload.get("content", ""),
                "source": h.payload.get("source", ""),
                "doc_id": h.payload.get("doc_id", ""),
                "chunk_index": h.payload.get("chunk_index", 0),
            }
            for h in hits
        ]

    def build_context(self, query: str, category_id: Optional[str] = None,
                      top_k: int = 5) -> str:
        """检索并拼成 prompt 上下文字符串"""
        results = self.search(query, category_id, top_k)
        if not results:
            return ""
        parts = [f"[{i+1}] {r['content']}" for i, r in enumerate(results)]
        return "\n\n".join(parts)


# 全局实例（懒加载，避免启动时 Qdrant 未就绪报错）
_knowledge_service: Optional[KnowledgeService] = None


def get_knowledge_service() -> KnowledgeService:
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeService()
    return _knowledge_service
