# colbert_engine.py
import hashlib
import uuid
from pylate import models as pylate_models
from qdrant_client import QdrantClient, models as q_models
import torch


class ColBERTEngine:
    def __init__(self, collection_name="krag", device="cuda"):
        # Connect to Qdrant (works across all OS)
        self.client = QdrantClient("http://localhost:6333")
        self.collection_name = collection_name

        # Initialize ModernColBERT
        self.model = pylate_models.ColBERT(
            model_name_or_path="lightonai/GTE-ModernColBERT-v1",
            device=device,
            document_length=8192
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates collection if it doesn't exist with MaxSim configuration."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "colbert": q_models.VectorParams(
                        size=128,
                        distance=q_models.Distance.COSINE,
                        multivector_config=q_models.MultiVectorConfig(
                            comparator=q_models.MultiVectorComparator.MAX_SIM
                        )
                    )
                }
            )

    def ingest_batches(self, documents_with_meta, batch_size=16):
        """
        Ingests documents into Qdrant.
        Uses deterministic UUIDs based on content to prevent duplicates.
        """
        for i in range(0, len(documents_with_meta), batch_size):
            batch = documents_with_meta[i: i + batch_size]

            # Encode the text chunks
            embeddings = self.model.encode([d.page_content for d in batch], is_query=False)

            points = []
            for j, emb in enumerate(embeddings):
                # Create a deterministic seed for the ID
                # Combining content + source ensures the same chunk always gets the same ID
                uid_seed = f"{batch[j].page_content}{batch[j].metadata.get('source', '')}"

                # Generate a stable UUID based on the MD5 hash of the seed
                # This is more robust than integer hashing across different Python sessions
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, hashlib.md5(uid_seed.encode()).hexdigest()))

                points.append(
                    q_models.PointStruct(
                        id=point_id,
                        vector={"colbert": emb.tolist()},
                        payload={
                            "text": batch[j].page_content,
                            **batch[j].metadata
                        }
                    )
                )

            # Upsert into Qdrant (Update if ID exists, otherwise Insert)
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def search(self, query, k=5):
        """Performs MaxSim retrieval using the query embedding."""
        query_emb = self.model.encode([query], is_query=True)[0].tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb,
            using="colbert",
            limit=k
        ).points
        return results

    def delete_by_source(self, source_path):
        """
        Wipes all points associated with a specific file path.
        Critical for keeping the DB clean when a file is modified or deleted.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=q_models.FilterSelector(
                filter=q_models.Filter(
                    must=[
                        q_models.FieldCondition(
                            key="source",
                            match=q_models.MatchValue(value=str(source_path))
                        )
                    ]
                )
            )
        )

    def get_points_count(self):
        """Helper to verify DB state."""
        return self.client.get_collection(self.collection_name).points_count