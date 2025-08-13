# from __future__ import annotations

# import threading
# import uuid
# from pathlib import Path
# from typing import Optional, Sequence, cast

# from theflow.settings import settings as flowsettings

# from kotaemon.base import BaseComponent, Document, RetrievedDocument
# from kotaemon.embeddings import BaseEmbeddings
# from kotaemon.storages import BaseDocumentStore, BaseVectorStore

# from .base import BaseIndexing, BaseRetrieval
# from .rankings import BaseReranking, LLMReranking

# VECTOR_STORE_FNAME = "vectorstore"
# DOC_STORE_FNAME = "docstore"


# class VectorIndexing(BaseIndexing):
#     """Ingest the document, run through the embedding, and store the embedding in a
#     vector store.

#     This pipeline supports the following set of inputs:
#         - List of documents
#         - List of texts
#     """

#     cache_dir: Optional[str] = getattr(flowsettings, "KH_CHUNKS_OUTPUT_DIR", None)
#     vector_store: BaseVectorStore
#     doc_store: Optional[BaseDocumentStore] = None
#     embedding: BaseEmbeddings
#     count_: int = 0

#     def to_retrieval_pipeline(self, *args, **kwargs):
#         """Convert the indexing pipeline to a retrieval pipeline"""
#         return VectorRetrieval(
#             vector_store=self.vector_store,
#             doc_store=self.doc_store,
#             embedding=self.embedding,
#             **kwargs,
#         )

#     def write_chunk_to_file(self, docs: list[Document]):
#         # save the chunks content into markdown format
#         if self.cache_dir:
#             file_name = docs[0].metadata.get("file_name")
#             if not file_name:
#                 return

#             file_name = Path(file_name)
#             for i in range(len(docs)):
#                 markdown_content = ""
#                 if "page_label" in docs[i].metadata:
#                     page_label = str(docs[i].metadata["page_label"])
#                     markdown_content += f"Page label: {page_label}"
#                 if "file_name" in docs[i].metadata:
#                     filename = docs[i].metadata["file_name"]
#                     markdown_content += f"\nFile name: {filename}"
#                 if "section" in docs[i].metadata:
#                     section = docs[i].metadata["section"]
#                     markdown_content += f"\nSection: {section}"
#                 if "type" in docs[i].metadata:
#                     if docs[i].metadata["type"] == "image":
#                         image_origin = docs[i].metadata["image_origin"]
#                         image_origin = f'<p><img src="{image_origin}"></p>'
#                         markdown_content += f"\nImage origin: {image_origin}"
#                 if docs[i].text:
#                     markdown_content += f"\ntext:\n{docs[i].text}"

#                 with open(
#                     Path(self.cache_dir) / f"{file_name.stem}_{self.count_+i}.md",
#                     "w",
#                     encoding="utf-8",
#                 ) as f:
#                     f.write(markdown_content)

#     def add_to_docstore(self, docs: list[Document]):
#         if self.doc_store:
#             print("Adding documents to doc store")
#             self.doc_store.add(docs)

#     def add_to_vectorstore(self, docs: list[Document]):
#         # in case we want to skip embedding
#         if self.vector_store:
#             print(f"Getting embeddings for {len(docs)} nodes")
#             embeddings = self.embedding(docs)
#             print("Adding embeddings to vector store")
#             self.vector_store.add(
#                 embeddings=embeddings,
#                 ids=[t.doc_id for t in docs],
#             )

#     def run(self, text: str | list[str] | Document | list[Document]):
#         input_: list[Document] = []
#         if not isinstance(text, list):
#             text = [text]

#         for item in cast(list, text):
#             if isinstance(item, str):
#                 input_.append(Document(text=item, id_=str(uuid.uuid4())))
#             elif isinstance(item, Document):
#                 input_.append(item)
#             else:
#                 raise ValueError(
#                     f"Invalid input type {type(item)}, should be str or Document"
#                 )

#         self.add_to_vectorstore(input_)
#         self.add_to_docstore(input_)
#         self.write_chunk_to_file(input_)
#         self.count_ += len(input_)


# class VectorRetrieval(BaseRetrieval):
#     """Retrieve list of documents from vector store"""

#     vector_store: BaseVectorStore
#     doc_store: Optional[BaseDocumentStore] = None
#     embedding: BaseEmbeddings
#     rerankers: Sequence[BaseReranking] = []
#     top_k: int = 5
#     first_round_top_k_mult: int = 10
#     retrieval_mode: str = "hybrid"  # vector, text, hybrid

#     def _filter_docs(
#         self, documents: list[RetrievedDocument], top_k: int | None = None
#     ):
#         if top_k:
#             documents = documents[:top_k]
#         return documents

#     def run(
#         self, text: str | Document, top_k: Optional[int] = None, **kwargs
#     ) -> list[RetrievedDocument]:
#         """Retrieve a list of documents from vector store

#         Args:
#             text: the text to retrieve similar documents
#             top_k: number of top similar documents to return

#         Returns:
#             list[RetrievedDocument]: list of retrieved documents
#         """
#         if top_k is None:
#             top_k = self.top_k

#         do_extend = kwargs.pop("do_extend", False)
#         thumbnail_count = kwargs.pop("thumbnail_count", 3)

#         if do_extend:
#             top_k_first_round = top_k * self.first_round_top_k_mult
#         else:
#             top_k_first_round = top_k

#         if self.doc_store is None:
#             raise ValueError(
#                 "doc_store is not provided. Please provide a doc_store to "
#                 "retrieve the documents"
#             )

#         result: list[RetrievedDocument] = []
#         # TODO: should declare scope directly in the run params
#         scope = kwargs.pop("scope", None)
#         emb: list[float]

#         if self.retrieval_mode == "vector":
#             emb = self.embedding(text)[0].embedding
#             _, scores, ids = self.vector_store.query(
#                 embedding=emb, top_k=top_k_first_round, doc_ids=scope, **kwargs
#             )
#             docs = self.doc_store.get(ids)
#             result = [
#                 RetrievedDocument(**doc.to_dict(), score=score)
#                 for doc, score in zip(docs, scores)
#             ]
#         elif self.retrieval_mode == "text":
#             query = text.text if isinstance(text, Document) else text
#             docs = []
#             if scope:
#                 docs = self.doc_store.query(
#                     query, top_k=top_k_first_round, doc_ids=scope
#                 )
#             result = [RetrievedDocument(**doc.to_dict(), score=-1.0) for doc in docs]
#         elif self.retrieval_mode == "hybrid":
#             from kotaemon.storages import ElasticsearchVectorStore
#             if isinstance(self.vector_store, ElasticsearchVectorStore):
#                 # TODO: implement using ElasticsearchVectorStore.as_retriever(search_kwargs={"k": 10}).invoke(text)
#                 # TODO: can implement in storages/elasticsearch.py hybrid search, then we can call in here
#                 pass
#             else:
#                 # similarity search section
#                 emb = self.embedding(text)[0].embedding
#                 vs_docs: list[RetrievedDocument] = []
#                 vs_ids: list[str] = []
#                 vs_scores: list[float] = []

#                 def query_vectorstore():
#                     nonlocal vs_docs
#                     nonlocal vs_scores
#                     nonlocal vs_ids

#                     assert self.doc_store is not None
#                     _, vs_scores, vs_ids = self.vector_store.query(
#                         embedding=emb, top_k=top_k_first_round, doc_ids=scope, **kwargs
#                     )
#                     if vs_ids:
#                         vs_docs = self.doc_store.get(vs_ids)

#                 # full-text search section
#                 ds_docs: list[RetrievedDocument] = []

#                 def query_docstore():
#                     nonlocal ds_docs

#                     assert self.doc_store is not None
#                     query = text.text if isinstance(text, Document) else text
#                     if scope:
#                         ds_docs = self.doc_store.query(
#                             query, top_k=top_k_first_round, doc_ids=scope
#                         )

#                 vs_query_thread = threading.Thread(target=query_vectorstore)
#                 ds_query_thread = threading.Thread(target=query_docstore)

#                 vs_query_thread.start()
#                 ds_query_thread.start()

#                 vs_query_thread.join()
#                 ds_query_thread.join()

#                 result = [
#                     RetrievedDocument(**doc.to_dict(), score=-1.0)
#                     for doc in ds_docs
#                     if doc not in vs_ids
#                 ]
#                 result += [
#                     RetrievedDocument(**doc.to_dict(), score=score)
#                     for doc, score in zip(vs_docs, vs_scores)
#                 ]
#                 print(f"Got {len(vs_docs)} from vectorstore")
#                 print(f"Got {len(ds_docs)} from docstore")

#         # use additional reranker to re-order the document list
#         if self.rerankers and text:
#             for reranker in self.rerankers:
#                 # if reranker is LLMReranking, limit the document with top_k items only
#                 if isinstance(reranker, LLMReranking):
#                     result = self._filter_docs(result, top_k=top_k)
#                 result = reranker.run(documents=result, query=text)

#         result = self._filter_docs(result, top_k=top_k)
#         print(f"Got raw {len(result)} retrieved documents")

#         # add page thumbnails to the result if exists
#         thumbnail_doc_ids: set[str] = set()
#         # we should copy the text from retrieved text chunk
#         # to the thumbnail to get relevant LLM score correctly
#         text_thumbnail_docs: dict[str, RetrievedDocument] = {}

#         non_thumbnail_docs = []
#         raw_thumbnail_docs = []
#         for doc in result:
#             if doc.metadata.get("type") == "thumbnail":
#                 # change type to image to display on UI
#                 doc.metadata["type"] = "image"
#                 raw_thumbnail_docs.append(doc)
#                 continue
#             if (
#                 "thumbnail_doc_id" in doc.metadata
#                 and len(thumbnail_doc_ids) < thumbnail_count
#             ):
#                 thumbnail_id = doc.metadata["thumbnail_doc_id"]
#                 thumbnail_doc_ids.add(thumbnail_id)
#                 text_thumbnail_docs[thumbnail_id] = doc
#             else:
#                 non_thumbnail_docs.append(doc)

#         linked_thumbnail_docs = self.doc_store.get(list(thumbnail_doc_ids))
#         print(
#             "thumbnail docs",
#             len(linked_thumbnail_docs),
#             "non-thumbnail docs",
#             len(non_thumbnail_docs),
#             "raw-thumbnail docs",
#             len(raw_thumbnail_docs),
#         )
#         additional_docs = []

#         for thumbnail_doc in linked_thumbnail_docs:
#             text_doc = text_thumbnail_docs[thumbnail_doc.doc_id]
#             doc_dict = thumbnail_doc.to_dict()
#             doc_dict["_id"] = text_doc.doc_id
#             doc_dict["content"] = text_doc.content
#             doc_dict["metadata"]["type"] = "image"
#             for key in text_doc.metadata:
#                 if key not in doc_dict["metadata"]:
#                     doc_dict["metadata"][key] = text_doc.metadata[key]

#             additional_docs.append(RetrievedDocument(**doc_dict, score=text_doc.score))

#         result = additional_docs + non_thumbnail_docs

#         if not result:
#             # return output from raw retrieved thumbnails
#             result = self._filter_docs(raw_thumbnail_docs, top_k=thumbnail_count)

#         return result


# class TextVectorQA(BaseComponent):
#     retrieving_pipeline: BaseRetrieval
#     qa_pipeline: BaseComponent

#     def run(self, question, **kwargs):
#         retrieved_documents = self.retrieving_pipeline(question, **kwargs)
#         return self.qa_pipeline(question, retrieved_documents, **kwargs)
from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Optional, Sequence, cast

from theflow.settings import settings as flowsettings

from kotaemon.base import BaseComponent, Document, RetrievedDocument
from kotaemon.embeddings import BaseEmbeddings
from kotaemon.storages import BaseDocumentStore, BaseVectorStore

from .base import BaseIndexing, BaseRetrieval
from .rankings import BaseReranking, LLMReranking

VECTOR_STORE_FNAME = "vectorstore"
DOC_STORE_FNAME = "docstore"


class VectorIndexing(BaseIndexing):
    """Ingest the document, run through the embedding, and store the embedding in a
    vector store.

    This pipeline supports the following set of inputs:
        - List of documents
        - List of texts
    """

    cache_dir: Optional[str] = getattr(flowsettings, "KH_CHUNKS_OUTPUT_DIR", None)
    vector_store: BaseVectorStore
    doc_store: Optional[BaseDocumentStore] = None
    embedding: BaseEmbeddings
    count_: int = 0

    def to_retrieval_pipeline(self, *args, **kwargs):
        """Convert the indexing pipeline to a retrieval pipeline"""
        return VectorRetrieval(
            vector_store=self.vector_store,
            doc_store=self.doc_store,
            embedding=self.embedding,
            **kwargs,
        )

    def write_chunk_to_file(self, docs: list[Document]):
        # save the chunks content into markdown format
        if self.cache_dir:
            file_name = docs[0].metadata.get("file_name")
            if not file_name:
                return

            file_name = Path(file_name)
            for i in range(len(docs)):
                markdown_content = ""
                if "page_label" in docs[i].metadata:
                    page_label = str(docs[i].metadata["page_label"])
                    markdown_content += f"Page label: {page_label}"
                if "file_name" in docs[i].metadata:
                    filename = docs[i].metadata["file_name"]
                    markdown_content += f"\nFile name: {filename}"
                if "section" in docs[i].metadata:
                    section = docs[i].metadata["section"]
                    markdown_content += f"\nSection: {section}"
                if "type" in docs[i].metadata:
                    if docs[i].metadata["type"] == "image":
                        image_origin = docs[i].metadata["image_origin"]
                        image_origin = f'<p><img src="{image_origin}"></p>'
                        markdown_content += f"\nImage origin: {image_origin}"
                if docs[i].text:
                    markdown_content += f"\ntext:\n{docs[i].text}"

                with open(
                    Path(self.cache_dir) / f"{file_name.stem}_{self.count_+i}.md",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(markdown_content)

    def add_to_docstore(self, docs: list[Document]):
        if self.doc_store:
            print("Adding documents to doc store")
            self.doc_store.add(docs)

    def add_to_vectorstore(self, docs: list[Document]):
        # in case we want to skip embedding
        if self.vector_store:
            print(f"Getting embeddings for {len(docs)} nodes")
            embeddings = self.embedding(docs)
            print("Adding embeddings to vector store")
            self.vector_store.add(
                embeddings=embeddings,
                ids=[t.doc_id for t in docs],
            )

    def run(self, text: str | list[str] | Document | list[Document]):
        input_: list[Document] = []
        if not isinstance(text, list):
            text = [text]

        for item in cast(list, text):
            if isinstance(item, str):
                input_.append(Document(text=item, id_=str(uuid.uuid4())))
            elif isinstance(item, Document):
                input_.append(item)
            else:
                raise ValueError(
                    f"Invalid input type {type(item)}, should be str or Document"
                )

        self.add_to_vectorstore(input_)
        self.add_to_docstore(input_)
        self.write_chunk_to_file(input_)
        self.count_ += len(input_)


class VectorRetrieval(BaseRetrieval):
    """Retrieve list of documents from vector store"""

    vector_store: BaseVectorStore
    doc_store: Optional[BaseDocumentStore] = None
    embedding: BaseEmbeddings
    rerankers: Sequence[BaseReranking] = []
    top_k: int = 5
    first_round_top_k_mult: int = 10
    retrieval_mode: str = "hybrid"  # vector, text, hybrid

    def _filter_docs(
        self, documents: list[RetrievedDocument], top_k: int | None = None
    ):
        if top_k:
            documents = documents[:top_k]
        return documents

    def _elasticsearch_hybrid_search(
        self, 
        text: str | Document, 
        top_k_first_round: int,
        scope: Optional[list[str]] = None,
        **kwargs
    ) -> list[RetrievedDocument]:
        """
        Perform hybrid search using ElasticsearchVectorStore's as_retriever method.
        
        Args:
            text: Query text or Document
            top_k_first_round: Number of results to retrieve
            scope: Optional list of document IDs to filter by
            **kwargs: Additional search parameters
            
        Returns:
            List of retrieved documents with scores
        """
        from kotaemon.storages import ElasticsearchVectorStore
        
        if not isinstance(self.vector_store, ElasticsearchVectorStore):
            raise ValueError("This method requires ElasticsearchVectorStore")
        
        # Extract query text
        query_text = text.text if isinstance(text, Document) else text
        
        # Prepare search kwargs for the retriever
        search_kwargs = {"k": top_k_first_round}
        
        # Add scope filtering if provided
        if scope:
            search_kwargs["filter"] = {"terms": {"_id": scope}}
        
        # Add any additional filters from kwargs
        if "filter" in kwargs:
            additional_filter = kwargs["filter"]
            if "filter" in search_kwargs:
                # Combine filters using bool query
                search_kwargs["filter"] = {
                    "bool": {
                        "must": [search_kwargs["filter"], additional_filter]
                    }
                }
            else:
                search_kwargs["filter"] = additional_filter
        
        # Add other search parameters from kwargs
        for key in ["fetch_k", "lambda_mult", "score_threshold"]:
            if key in kwargs:
                search_kwargs[key] = kwargs[key]
        print(f"{search_kwargs = }")
        
        try:
            # Use ElasticsearchVectorStore's as_retriever method as specified in TODO
            retriever = self.vector_store._es_store.as_retriever(search_kwargs=search_kwargs)
            langchain_docs = retriever.invoke(query_text)
            
            # Convert LangChain Documents to RetrievedDocuments
            result = []
            for i, doc in enumerate(langchain_docs):
                # Extract document ID from metadata or generate one
                doc_id = doc.metadata.get('_id') or doc.metadata.get('id') or str(uuid.uuid4())
                
                # Since as_retriever doesn't return scores, we'll use rank-based scoring
                # Higher rank (earlier in results) gets higher score
                rank_score = 1.0 - (i / len(langchain_docs)) if langchain_docs else 0.0
                
                # Create RetrievedDocument
                retrieved_doc = RetrievedDocument(
                    doc_id=doc_id,
                    text=doc.page_content,
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=rank_score
                )
                result.append(retrieved_doc)
            
            print(f"Elasticsearch retriever returned {len(result)} documents")
            return result
            
        except Exception as e:
            print(f"Elasticsearch as_retriever failed: {e}")
            # Fallback to manual hybrid approach
            return self._manual_hybrid_search(text, top_k_first_round, scope, **kwargs)
    
    def _manual_hybrid_search(
        self, 
        text: str | Document, 
        top_k_first_round: int,
        scope: Optional[list[str]] = None,
        **kwargs
    ) -> list[RetrievedDocument]:
        """
        Fallback manual hybrid search implementation using separate vector and text searches.
        """
        if self.doc_store is None:
            raise ValueError("doc_store is required for manual hybrid search")
        
        # Get embedding for vector search
        emb = self.embedding(text)[0].embedding
        
        # Initialize result containers
        vs_docs: list[RetrievedDocument] = []
        vs_ids: list[str] = []
        vs_scores: list[float] = []
        ds_docs: list[Document] = []

        def query_vectorstore():
            nonlocal vs_docs, vs_scores, vs_ids
            _, vs_scores, vs_ids = self.vector_store.query(
                embedding=emb, top_k=top_k_first_round, doc_ids=scope, **kwargs
            )
            if vs_ids and self.doc_store:
                vs_docs = self.doc_store.get(vs_ids)

        def query_docstore():
            nonlocal ds_docs
            query = text.text if isinstance(text, Document) else text
            if scope and self.doc_store:
                ds_docs = self.doc_store.query(
                    query, top_k=top_k_first_round, doc_ids=scope
                )

        # Run searches in parallel
        vs_query_thread = threading.Thread(target=query_vectorstore)
        ds_query_thread = threading.Thread(target=query_docstore)

        vs_query_thread.start()
        ds_query_thread.start()
        vs_query_thread.join()
        ds_query_thread.join()

        # Combine results, avoiding duplicates
        result = []
        
        # Add text search results first (with default score)
        text_doc_ids = {doc.doc_id for doc in ds_docs}
        for doc in ds_docs:
            if doc.doc_id not in vs_ids:  # Avoid duplicates
                result.append(RetrievedDocument(**doc.to_dict(), score=-1.0))
        
        # Add vector search results with their scores
        for doc, score in zip(vs_docs, vs_scores):
            result.append(RetrievedDocument(**doc.to_dict(), score=score))
        
        print(f"Manual hybrid: {len(vs_docs)} from vectorstore, {len(ds_docs)} from docstore")
        return result

    def run(
        self, text: str | Document, top_k: Optional[int] = None, **kwargs
    ) -> list[RetrievedDocument]:
        """Retrieve a list of documents from vector store

        Args:
            text: the text to retrieve similar documents
            top_k: number of top similar documents to return

        Returns:
            list[RetrievedDocument]: list of retrieved documents
        """
        if top_k is None:
            top_k = self.top_k

        do_extend = kwargs.pop("do_extend", False)
        thumbnail_count = kwargs.pop("thumbnail_count", 3)

        if do_extend:
            top_k_first_round = top_k * self.first_round_top_k_mult
        else:
            top_k_first_round = top_k

        if self.doc_store is None:
            raise ValueError(
                "doc_store is not provided. Please provide a doc_store to "
                "retrieve the documents"
            )

        result: list[RetrievedDocument] = []
        scope = kwargs.pop("scope", None)

        if self.retrieval_mode == "vector":
            emb = self.embedding(text)[0].embedding
            _, scores, ids = self.vector_store.query(
                embedding=emb, top_k=top_k_first_round, doc_ids=scope, **kwargs
            )
            docs = self.doc_store.get(ids)
            result = [
                RetrievedDocument(**doc.to_dict(), score=score)
                for doc, score in zip(docs, scores)
            ]
        elif self.retrieval_mode == "text":
            query = text.text if isinstance(text, Document) else text
            docs = []
            if scope:
                docs = self.doc_store.query(
                    query, top_k=top_k_first_round, doc_ids=scope
                )
            result = [RetrievedDocument(**doc.to_dict(), score=-1.0) for doc in docs]
        elif self.retrieval_mode == "hybrid":
            from kotaemon.storages import ElasticsearchVectorStore
            if isinstance(self.vector_store, ElasticsearchVectorStore):
                # Use ElasticsearchVectorStore's built-in hybrid search capabilities
                result = self._elasticsearch_hybrid_search(
                    text, top_k_first_round, scope, **kwargs
                )
            else:
                # Fallback to manual hybrid search for other vector stores
                result = self._manual_hybrid_search(
                    text, top_k_first_round, scope, **kwargs
                )

        # use additional reranker to re-order the document list
        if self.rerankers and text:
            for reranker in self.rerankers:
                # if reranker is LLMReranking, limit the document with top_k items only
                if isinstance(reranker, LLMReranking):
                    result = self._filter_docs(result, top_k=top_k)
                result = reranker.run(documents=result, query=text)

        result = self._filter_docs(result, top_k=top_k)
        print(f"Got raw {len(result)} retrieved documents")

        # add page thumbnails to the result if exists
        thumbnail_doc_ids: set[str] = set()
        # we should copy the text from retrieved text chunk
        # to the thumbnail to get relevant LLM score correctly
        text_thumbnail_docs: dict[str, RetrievedDocument] = {}

        non_thumbnail_docs = []
        raw_thumbnail_docs = []
        for doc in result:
            if doc.metadata.get("type") == "thumbnail":
                # change type to image to display on UI
                doc.metadata["type"] = "image"
                raw_thumbnail_docs.append(doc)
                continue
            if (
                "thumbnail_doc_id" in doc.metadata
                and len(thumbnail_doc_ids) < thumbnail_count
            ):
                thumbnail_id = doc.metadata["thumbnail_doc_id"]
                thumbnail_doc_ids.add(thumbnail_id)
                text_thumbnail_docs[thumbnail_id] = doc
            else:
                non_thumbnail_docs.append(doc)

        linked_thumbnail_docs = self.doc_store.get(list(thumbnail_doc_ids))
        print(
            "thumbnail docs",
            len(linked_thumbnail_docs),
            "non-thumbnail docs",
            len(non_thumbnail_docs),
            "raw-thumbnail docs",
            len(raw_thumbnail_docs),
        )
        additional_docs = []

        for thumbnail_doc in linked_thumbnail_docs:
            text_doc = text_thumbnail_docs[thumbnail_doc.doc_id]
            doc_dict = thumbnail_doc.to_dict()
            doc_dict["_id"] = text_doc.doc_id
            doc_dict["content"] = text_doc.content
            doc_dict["metadata"]["type"] = "image"
            for key in text_doc.metadata:
                if key not in doc_dict["metadata"]:
                    doc_dict["metadata"][key] = text_doc.metadata[key]

            additional_docs.append(RetrievedDocument(**doc_dict, score=text_doc.score))

        result = additional_docs + non_thumbnail_docs

        if not result:
            # return output from raw retrieved thumbnails
            result = self._filter_docs(raw_thumbnail_docs, top_k=thumbnail_count)

        return result


class TextVectorQA(BaseComponent):
    retrieving_pipeline: BaseRetrieval
    qa_pipeline: BaseComponent

    def run(self, question, **kwargs):
        retrieved_documents = self.retrieving_pipeline(question, **kwargs)
        return self.qa_pipeline(question, retrieved_documents, **kwargs)