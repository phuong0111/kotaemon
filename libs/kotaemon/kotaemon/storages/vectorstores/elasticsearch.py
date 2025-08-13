from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, cast

from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import OllamaEmbeddings

from kotaemon.base import DocumentWithEmbedding
from .base import LlamaIndexVectorStore


class ElasticsearchVectorStore(LlamaIndexVectorStore):
    """Elasticsearch vector store implementation for kotaemon"""
    
    def __init__(
        self,
        elastic_endpoint: str = "http://localhost:9200",
        index_name: str = "kotaemon",
        embedding_model: str = "nomic-embed-text:latest",
        username: str = "elastic",
        password: str = "mta2024",
        distance_strategy: str = "COSINE",
        top_k: int = 10,
        hybrid: bool = True,
        **kwargs: Any,
    ):
        """Initialize Elasticsearch vector store
        
        Args:
            elastic_endpoint: Elasticsearch endpoint (default: "http://localhost:9200")
            index_name: Name of the Elasticsearch index (default: "kotaemon")
            embedding_model: Name of the embedding model to use (default: "nomic-embed-text:latest")
            username: Elasticsearch username (default: "elastic")
            password: Elasticsearch password (default: "mta2024")
            distance_strategy: Distance strategy for similarity search (default: "COSINE")
            top_k: Default number of results to retrieve (default: 10)
            hybrid: Whether to use hybrid search (vector + text) (default: True)
            **kwargs: Additional arguments passed to ElasticsearchStore
        """
        logging.info(">> Elasticsearch connection setup!")
        
        self._elastic_endpoint = elastic_endpoint
        self._index_name = index_name
        self._embedding_model = embedding_model
        self._username = username
        self._password = password
        self._distance_strategy = distance_strategy
        self._top_k = top_k
        self._hybrid = hybrid
        self._kwargs = kwargs
        
        # Setup Elasticsearch client
        self._elastic_host = elastic_endpoint
        if not elastic_endpoint.startswith('http'):
            self._elastic_host = f"http://{username}:{password}@{elastic_endpoint}:9200"
        
        self._elastic_client = Elasticsearch(
            [self._elastic_host], 
            basic_auth=(username, password), 
            http_compress=True
        )
        
        if not self._elastic_client.ping():
            raise ValueError("Connection to Elasticsearch failed")
        
        # Setup embeddings
        self._embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Setup ElasticsearchStore
        from langchain_elasticsearch import ElasticsearchStore
        
        strategy = ElasticsearchStore.ApproxRetrievalStrategy(hybrid=hybrid)
        
        self._es_store = ElasticsearchStore(
            es_connection=self._elastic_client,
            embedding=self._embeddings,
            index_name=self._index_name,
            distance_strategy=distance_strategy,
            strategy=strategy,
            # **kwargs
        )
        
        # Don't call super().__init__() since we're not using LlamaIndex directly
        # Instead, we'll implement the interface methods ourselves
        
    def add(
        self,
        embeddings: list[list[float]] | list[DocumentWithEmbedding],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ):
        if isinstance(embeddings[0], list):
            nodes: list[DocumentWithEmbedding] = [
                DocumentWithEmbedding(embedding=embedding) for embedding in embeddings
            ]
        else:
            nodes = embeddings  # type: ignore
        if metadatas is not None:
            for node, metadata in zip(nodes, metadatas):
                node.metadata = metadata
        if ids is not None:
            for node, id in zip(nodes, ids):
                node.id_ = id
        
        # Convert nodes to texts and metadatas for ElasticsearchStore
        texts = []
        final_metadatas = []
        final_ids = []
        embeddings = []
        
        for node in nodes:
            texts.append(node.text or str(node.content))
            embeddings.append(node.embedding)
            metadata = node.metadata.copy() if node.metadata else {}
            final_metadatas.append(metadata)
            final_ids.append(node.id_)
        
        return self._es_store.add_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            metadatas=final_metadatas,
            ids=final_ids
        )
    
    def delete(self, ids: list[str], **kwargs):
        """Delete documents from Elasticsearch
        
        Args:
            ids: List of document IDs to delete
        """
        # ElasticsearchStore doesn't have a direct delete method
        # We need to use the underlying Elasticsearch client
        for doc_id in ids:
            try:
                self._elastic_client.delete(
                    index=self._index_name,
                    id=doc_id,
                    ignore=[404]  # Ignore if document doesn't exist
                )
            except Exception as e:
                logging.warning(f"Failed to delete document {doc_id}: {e}")
    
    def query(
        self,
        embedding: list[float],
        top_k: int = 1,
        ids: Optional[list[str]] = None,
        **kwargs,
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """Query Elasticsearch for similar vectors using ElasticsearchStore infrastructure
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            ids: Optional list of document IDs to filter by
            **kwargs: Extra query parameters (filter, fetch_k, etc.)
            
        Returns:
            Tuple of (embeddings, similarities, ids)
        """
        # Check if index exists
        if not self._elastic_client.indices.exists(index=self._index_name):
            return [], [], []
        
        try:
            # Build KNN query for vector search
            knn_query = {
                "field": "vector",
                "query_vector": embedding,
                "k": top_k,
                "num_candidates": kwargs.get('fetch_k', top_k * 10)
            }
            
            # Build the search body
            search_body = {
                "knn": knn_query,
                "_source": True,
                "size": top_k
            }
            
            # Handle filtering
            filters = []
            
            # Add ID filter if provided
            if ids is not None:
                filters.append({"terms": {"_id": ids}})
            
            # Add custom filter from kwargs
            custom_filter = kwargs.get('filter')
            if custom_filter:
                if isinstance(custom_filter, list):
                    filters.extend(custom_filter)
                else:
                    filters.append(custom_filter)
            
            # Add filters to the query if any exist
            if filters:
                if len(filters) == 1:
                    search_body["query"] = {"bool": {"filter": filters[0]}}
                else:
                    search_body["query"] = {"bool": {"filter": filters}}
            
            # Execute the search using ElasticsearchStore's client
            response = self._es_store.client.search(
                index=self._index_name,
                body=search_body
            )
            
            if not response.get('hits', {}).get('hits'):
                return [], [], []
            
            # Process results
            embeddings_out = []
            similarities = []
            result_ids = []
            
            for hit in response['hits']['hits']:
                source = hit['_source']
                
                # Get the stored vector embedding
                doc_embedding = source.get('vector')
                if doc_embedding is None:
                    # Fallback: compute embedding from text
                    text_content = source.get('text', '')
                    doc_embedding = self._embeddings.embed_query(text_content)
                
                embeddings_out.append(doc_embedding)
                
                # For hybrid search, scores may not be meaningful, so use rank-based scoring
                if self._hybrid:
                    # Use rank-based score (1.0 for first result, decreasing)
                    rank_score = 1.0 - (len(similarities) / top_k)
                    similarities.append(max(0.1, rank_score))  # Ensure minimum score
                else:
                    # Use actual Elasticsearch score for pure vector search
                    similarities.append(float(hit['_score']))
                
                # Get document ID
                doc_id = hit['_id']
                result_ids.append(str(doc_id))
            
            return embeddings_out, similarities, result_ids
            
        except Exception as e:
            logging.error(f"Elasticsearch query failed: {e}")
            return [], [], []
    
    def drop(self):
        """Drop the entire Elasticsearch index"""
        try:
            if self._elastic_client.indices.exists(index=self._index_name):
                self._elastic_client.indices.delete(index=self._index_name)
                logging.info(f"Dropped index: {self._index_name}")
            else:
                logging.info(f"Index {self._index_name} does not exist, nothing to drop")
        except Exception as e:
            logging.error(f"Failed to drop index {self._index_name}: {e}")
    
    def count(self) -> int:
        """Get the number of documents in the index"""
        try:
            # Check if index exists first
            if not self._elastic_client.indices.exists(index=self._index_name):
                return 0
            response = self._elastic_client.count(index=self._index_name)
            return response['count']
        except Exception as e:
            logging.error(f"Failed to count documents: {e}")
            return 0
    
    def __persist_flow__(self):
        """Return configuration for persistence"""
        return {
            "elastic_endpoint": self._elastic_endpoint,
            "index_name": self._index_name,
            "embedding_model": self._embedding_model,
            "username": self._username,
            "password": self._password,
            "distance_strategy": self._distance_strategy,
            "top_k": self._top_k,
            "hybrid": self._hybrid,
            **self._kwargs,
        }