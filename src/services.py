# Qdrant client setup
# Google Cloud Storage functions
# Gemini API calls
# Database connections


import torch
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from google.cloud import storage
from google import genai
from google.genai import types
from colpali_engine.models import ColQwen2, ColQwen2Processor
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.memory import MemorySaver
from langchain.embeddings import init_embeddings
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console

from .config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_TIMEOUT,
    MODEL_NAME, GEMINI_API_KEY, POSTGRES_CONNECTION_STRING,
    SEARCH_LIMIT, PREFETCH_LIMIT, LLM_CATEGORY
)

console = Console()

class QdrantService:
    """Service for managing Qdrant vector database operations"""
    
    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT
        )
        self.collection_name = QDRANT_COLLECTION_NAME
        self._verify_collection()
    
    def _verify_collection(self):
        """Verify if the collection exists"""
        try:
            self.client.get_collection(collection_name=self.collection_name)
            console.print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            if "Collection not found" in str(e):
                console.print(f"collection '{self.collection_name}' Not found")
            else:
                console.print(f"Error accessing Qdrant collection: {e}")
    
    def reranking_search_batch(self, query_batch, search_limit=SEARCH_LIMIT, prefetch_limit=PREFETCH_LIMIT, filter_list=LLM_CATEGORY):
        """Perform reranking search with batch queries"""
        filter_ = None
        if filter_list:
            filter_ = models.Filter(
                should=[
                    models.FieldCondition(
                        key="Book_Category",
                        match=models.MatchAny(any=filter_list)
                    )
                ]
            )

        search_queries = [
            models.QueryRequest(
                query=query,
                prefetch=[
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_columns"
                    ),
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_rows"
                    ),
                ],
                filter=filter_,
                limit=search_limit,
                with_payload=True,
                with_vector=False,
                using="original"
            ) for query in query_batch
        ]

        try:
            response = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=search_queries
            )
        except Exception as e:
            console.print(f"[Qdrant Error] Failed to query Qdrant with filter: {str(e)}")
            return f"Error querying Qdrant: {str(e)}"

        # Fallback search without filter if no results
        if all(not res.points for res in response):
            search_queries = [
                models.QueryRequest(
                    query=query,
                    prefetch=[
                        models.Prefetch(
                            query=query,
                            limit=prefetch_limit,
                            using="mean_pooling_columns"
                        ),
                        models.Prefetch(
                            query=query,
                            limit=prefetch_limit,
                            using="mean_pooling_rows"
                        ),
                    ],
                    filter=None,
                    limit=search_limit,
                    with_payload=True,
                    with_vector=False,
                    using="original"
                ) for query in query_batch
            ]
            try:
                response = self.client.query_batch_points(
                    collection_name=self.collection_name,
                    requests=search_queries
                )
            except Exception as e:
                console.print(f"[Qdrant Error] Failed to query Qdrant (fallback): {str(e)}")
                return f"Error querying Qdrant (fallback): {str(e)}"

        return response

class ModelService:
    """Service for managing AI models"""
    
    def __init__(self):
        console.print("Loading the model...")
        self.model = ColQwen2.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(MODEL_NAME, use_fast=True)
    
    def batch_embed_query(self, query_batch):
        """Batch embed queries using the model"""
        with torch.no_grad():
            processed_queries = self.processor.process_queries(query_batch).to(self.model.device)
            query_embeddings_batch = self.model(**processed_queries)
        return query_embeddings_batch.cpu().float().numpy()

class GCSService:
    """Service for Google Cloud Storage operations"""
    
    def __init__(self):
        self.client = storage.Client()
    
    def download_gcs_image(self, gcs_uri):
        """Download image from Google Cloud Storage"""
        try:
            if not gcs_uri.startswith("gs://"):
                raise ValueError("Invalid GCS URI format")
            bucket_name = gcs_uri.split("/")[2]
            blob_path = "/".join(gcs_uri.split("/")[3:])
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            return blob.download_as_bytes()
        except Exception as e:
            return f"Error downloading GCS image: {str(e)}"

class GeminiService:
    """Service for Google Gemini API operations"""
    
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
    
    def generate_content(self, prompt_text, images_payload):
        """Generate content using Gemini Vision API"""
        try:
            console.print("Sending data to Gemini Vision API...")
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt_text)] + images_payload)]
            )
            return response.text
        except Exception as e:
            console.print(f"GEMINI Error: {e}")
            return None

class DatabaseService:
    """Service for database operations"""
    
    @staticmethod
    def get_memory_checkpointer():
        """Get memory checkpointer for agent"""
        return MemorySaver()
    
    @staticmethod
    def get_embedder():
        """Get embeddings model"""
        return init_embeddings("openai:text-embedding-3-small")
    
    @staticmethod
    def get_profile_store():
        """Get profile store"""
        return InMemoryStore()
    
    @staticmethod
    async def get_async_postgres_store():
        """Get async PostgreSQL store"""
        return AsyncPostgresStore.from_conn_string(POSTGRES_CONNECTION_STRING)

# Initialize services
qdrant_service = QdrantService()
model_service = ModelService()
gcs_service = GCSService()
gemini_service = GeminiService()
database_service = DatabaseService()