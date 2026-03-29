# colbert_engine.py
import hashlib
import uuid
from typing import List, Any, Optional
from pylate import models as pylate_models
from qdrant_client import QdrantClient, models as q_models
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import ConfigDict, Field  # Make sure to import ConfigDict
import os
import torch
from colorama import Fore, Style, init

init(autoreset=True)


class ColBERTRetriever(BaseRetriever):
    engine: Any
    k: int = 5

    # This tells Pydantic to ignore the cyfunction that Cython creates
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=(type(lambda: None),),  # This usually catches cyfunctions
    )

    # ADD THIS: This is the entry point LangChain actually uses
    def invoke(
        self, input: str, config: Optional[Any] = None, **kwargs: Any
    ) -> List[Document]:
        print(f"{Fore.CYAN}[INVOKE] Manual hook triggered for query: {input}")
        return self._get_relevant_documents(input, **kwargs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        print(f"{Fore.YELLOW}[DEBUG] Inside _get_relevant_documents for: {query}")

        # Ensure search is actually called
        raw_results = self.engine.search(query, k=self.k)

        if not raw_results:
            print(f"{Fore.RED}[DEBUG] Engine search returned NOTHING")
            return []

        python_friendly_docs = []
        for res in raw_results:
            # Cython-safe payload access
            p_load = getattr(res, "payload", {}) if hasattr(res, "payload") else {}

            doc = Document(
                page_content=str(p_load.get("text", "No content")),
                metadata={k: v for k, v in p_load.items() if k != "text"},
            )
            python_friendly_docs.append(doc)

        print(
            f"{Fore.GREEN}[DEBUG] Returning {len(python_friendly_docs)} docs to chain."
        )
        return python_friendly_docs


class ColBERTEngine:
    def __init__(self, collection_name, device="cuda"):
        # Connect to Qdrant
        q_host = os.getenv("QDRANT_HOST", "localhost")
        self.client = QdrantClient(host=q_host, port=6333)

        # This now receives 'krag_dev' or 'krag_prod' from CollegeRAG
        self.collection_name = collection_name

        # Initialize ModernColBERT
        self.model = pylate_models.ColBERT(
            model_name_or_path="lightonai/GTE-ModernColBERT-v1",
            device=device,
            document_length=8192,
        )

        print(
            f"{Fore.MAGENTA} Device checking from colbert: {'GPU' if torch.cuda.is_available() else 'CPU'}"
        )

        # Critical: Set up the specific isolated collection
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates collection if it doesn't exist with MaxSim configuration."""
        # Using try-except or existence check is fine,
        # but this ensures the specific name is created
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating isolated collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "colbert": q_models.VectorParams(
                        size=128,
                        distance=q_models.Distance.COSINE,
                        multivector_config=q_models.MultiVectorConfig(
                            comparator=q_models.MultiVectorComparator.MAX_SIM
                        ),
                    )
                },
            )

    def ingest_batches(self, documents_with_meta, batch_size=16):
        for i in range(0, len(documents_with_meta), batch_size):
            batch = documents_with_meta[i : i + batch_size]

            # 1. Keep the GPU Busy (Fastest Step)
            embeddings = self.model.encode(
                [d.page_content for d in batch], is_query=False
            )

            # 2. Prepare the points
            all_points = []
            for j, emb in enumerate(embeddings):
                uid_seed = (
                    f"{batch[j].page_content}{batch[j].metadata.get('source', '')}"
                )
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS, hashlib.md5(uid_seed.encode()).hexdigest()
                    )
                )
                all_points.append(
                    q_models.PointStruct(
                        id=point_id,
                        vector={"colbert": emb.tolist()},
                        payload={"text": batch[j].page_content, **batch[j].metadata},
                    )
                )

            # 3. SMART UPLOAD (The Professional Standard)
            # We process in 16, but we upload in sub-chunks of 4 to avoid the 32MB limit.
            upload_chunk_size = 4
            for k in range(0, len(all_points), upload_chunk_size):
                sub_batch = all_points[k : k + upload_chunk_size]
                self.client.upsert(
                    collection_name=self.collection_name, points=sub_batch
                )

    def search(self, query, k=5):
        print("Searching in colbert database")
        """Performs MaxSIM retrieval using the query embedding."""
        query_emb = self.model.encode([query], is_query=True)[0].tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb,
            using="colbert",
            limit=k,
        ).points

        print(f"length of results: {len(results)}")

        return list(results)

    def as_retriever(self, search_kwargs=None):
        """Returns the LangChain-compatible retriever object."""
        k = (search_kwargs or {}).get("k", 5)
        return ColBERTRetriever(engine=self, k=k)

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
                            match=q_models.MatchValue(value=str(source_path)),
                        )
                    ]
                )
            ),
        )

    def get_points_count(self):
        """Helper to verify DB state."""
        return self.client.get_collection(self.collection_name).points_count


# --- THE FINAL PATCH ---

# 1. We re-link the method to ensure the C-compiled version is recognized
ColBERTRetriever._get_relevant_documents = ColBERTRetriever._get_relevant_documents
ColBERTRetriever._aget_relevant_documents = ColBERTRetriever._aget_relevant_documents

# 2. We manually remove the "Missing" labels from the class registry
ColBERTRetriever.__abstractmethods__ = frozenset(
    [
        m
        for m in ColBERTRetriever.__abstractmethods__
        if m not in ("_get_relevant_documents", "_aget_relevant_documents")
    ]
)
