"""Quản lý Qdrant Hybrid Collection (Dense + Sparse) và Payload Indexing."""

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from config import DENSE_VECTOR_SIZE, QDRANT_COLLECTION_NAME, UPSERT_BATCH_SIZE
from rag.indexing.embedder import DenseEmbedder, SparseBM25Encoder
from rag.schemas import ChunkMetadata

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME: str = "dense"
SPARSE_VECTOR_NAME: str = "sparse"


class QdrantHybridStore:
    """Kho lưu trữ vector lai (Dense + Sparse BM25) trên Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        dense_embedder: DenseEmbedder,
        sparse_encoder: SparseBM25Encoder | None = None,
        collection_name: str = QDRANT_COLLECTION_NAME,
    ) -> None:
        self.client = client
        self.dense_embedder = dense_embedder
        self.sparse_encoder = sparse_encoder or SparseBM25Encoder()
        self.collection_name = collection_name

    def ensure_collection(self) -> None:
        """Tạo collection Hybrid và đánh Payload Index nếu chưa tồn tại."""
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=self.dense_embedder.vector_size or DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                )
            },
        )
        self._create_payload_indexes()
        logger.info("Đã khởi tạo Qdrant Hybrid Collection '%s'.", self.collection_name)

    def _create_payload_indexes(self) -> None:
        """Đánh chỉ mục các trường lọc Metadata để Pre-filtering đạt tốc độ tối đa."""
        index_fields: dict[str, PayloadSchemaType] = {
            "ticker": PayloadSchemaType.KEYWORD,
            "year": PayloadSchemaType.INTEGER,
            "quarter": PayloadSchemaType.INTEGER,
            "doc_type": PayloadSchemaType.KEYWORD,
        }
        for field_name, schema_type in index_fields.items():
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )

    def upsert_chunks(
        self,
        chunks: list[ChunkMetadata],
        batch_size: int = UPSERT_BATCH_SIZE,
    ) -> int:
        """Mã hóa và nạp danh sách ChunkMetadata vào Qdrant theo từng lô."""
        if not chunks:
            raise ValueError("Danh sách chunks cần nạp không được rỗng.")

        self.ensure_collection()
        total_upserted = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            points = self._build_points(batch)
            self.client.upsert(collection_name=self.collection_name, points=points)
            total_upserted += len(points)
            logger.info("Đã nạp %d/%d chunks vào Qdrant.", total_upserted, len(chunks))

        return total_upserted

    def _build_points(self, batch: list[ChunkMetadata]) -> list[PointStruct]:
        """Tạo danh sách PointStruct gồm Dense vector, Sparse vector và Payload."""
        texts = [c.text for c in batch]
        dense_vectors = self.dense_embedder.embed_documents(texts)

        points: list[PointStruct] = []
        for chunk, dense_vec in zip(batch, dense_vectors, strict=True):
            sparse_vec = self.sparse_encoder.encode(chunk.text)
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                PointStruct(
                    id=point_uuid,
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: sparse_vec,
                    },
                    payload=chunk.model_dump(mode="json"),
                )
            )
        return points
