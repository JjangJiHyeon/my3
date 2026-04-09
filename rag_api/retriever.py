"""Minimal Chroma retriever wrapper."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any

from .config import RagApiConfig, load_latest_chroma_index, resolve_chroma_persist_dir, resolve_collection_name
from .observability import optional_traceable, stage_timer
from .summary_ranker import rerank_summary_documents


SUMMARY_MIN_COVERAGE_CHUNKS = 24
SUMMARY_MAX_COVERAGE_CHUNKS = 60
SUMMARY_COVERAGE_CACHE_SIZE = 64
SUMMARY_SELECTION_CACHE_SIZE = 128

_summary_cache_lock = Lock()
_summary_coverage_cache: OrderedDict[tuple[str, str, str, str], tuple[dict[str, Any], ...]] = OrderedDict()
_summary_selection_cache: OrderedDict[tuple[str, str, str, str, int], tuple[str, ...]] = OrderedDict()


@dataclass
class ChromaRetriever:
    config: RagApiConfig

    @optional_traceable(name="retriever.retrieve", run_type="retriever")
    def retrieve(
        self,
        query: str,
        strategy_name: str,
        top_k: int,
        filename_filter: str | None = None,
        request_id: str | None = None,
    ) -> list[Any]:
        with stage_timer(
            self.config.run_dir,
            "retriever.retrieve",
            request_id=request_id,
            strategy_name=strategy_name,
            top_k=top_k,
            filename_filter=filename_filter,
            query_len=len(query),
        ) as event:
            try:
                retriever = self._as_langchain_retriever(
                    top_k=top_k,
                    strategy_name=strategy_name,
                    filename_filter=filename_filter,
                    request_id=request_id,
                )
                if hasattr(retriever, "invoke"):
                    documents = list(retriever.invoke(query))
                else:
                    documents = list(retriever.get_relevant_documents(query))
                event["backend"] = "langchain_retriever"
                event["returned_docs"] = len(documents)
                return documents
            except RuntimeError as exc:
                if "Chroma retrieval" not in str(exc):
                    raise
                documents = self._native_chroma_retrieve(
                    query=query,
                    top_k=top_k,
                    strategy_name=strategy_name,
                    filename_filter=filename_filter,
                    include_scores=False,
                    request_id=request_id,
                )
                event["backend"] = "native_chroma_fallback"
                event["returned_docs"] = len(documents)
                return documents

    @optional_traceable(name="retriever.retrieve_representative", run_type="retriever")
    def retrieve_representative(
        self,
        filename: str,
        strategy_name: str,
        top_k: int,
        request_id: str | None = None,
    ) -> list[Any]:
        with stage_timer(
            self.config.run_dir,
            "retriever.retrieve_representative",
            request_id=request_id,
            filename=filename,
            strategy_name=strategy_name,
            top_k=top_k,
        ) as event:
            run_id, collection_name = self._summary_cache_namespace()
            max_chunks = _summary_coverage_budget(top_k)
            try:
                coverage_candidates = self._retrieve_summary_coverage_candidates(
                    strategy_name=strategy_name,
                    filename_filter=filename,
                    request_id=request_id,
                )
            except Exception:
                coverage_candidates = []
            event["coverage_candidates"] = len(coverage_candidates)
            if coverage_candidates:
                selection_key = _summary_selection_cache_key(
                    run_id=run_id,
                    collection_name=collection_name,
                    strategy_name=strategy_name,
                    filename_filter=filename,
                    max_chunks=max_chunks,
                )
                cached_doc_keys = _summary_selection_cache_get(selection_key)
                if cached_doc_keys:
                    documents = _restore_selected_documents(coverage_candidates, cached_doc_keys)
                    if documents:
                        event["backend"] = "coverage_selection_cache"
                        event["selection_cache_hit"] = True
                        event["returned_docs"] = len(documents)
                        return documents

                documents = _select_summary_coverage_documents(
                    coverage_candidates,
                    max_chunks=max_chunks,
                )
                _summary_selection_cache_put(
                    selection_key,
                    tuple(_document_key(document) for document in documents),
                )
                event["backend"] = "coverage_selection"
                event["selection_cache_hit"] = False
                event["returned_docs"] = len(documents)
                return documents

            query = _summary_representative_query(filename)
            candidate_k = min(80, max(top_k * 4, top_k + 12))
            candidates = self._retrieve_summary_candidates(
                query=query,
                strategy_name=strategy_name,
                top_k=candidate_k,
                filename_filter=filename,
                request_id=request_id,
            )
            documents = rerank_summary_documents(candidates, top_k=top_k)
            event["backend"] = "rerank_candidates"
            event["candidate_docs"] = len(candidates)
            event["returned_docs"] = len(documents)
            return documents

    def _summary_cache_namespace(self) -> tuple[str, str]:
        latest = load_latest_chroma_index(self.config)
        run_id = str(latest.get("run_id") or "")
        collection_name = str(resolve_collection_name(latest) or "")
        return run_id, collection_name

    def _as_langchain_retriever(
        self,
        top_k: int,
        strategy_name: str,
        filename_filter: str | None,
        request_id: str | None = None,
    ) -> Any:
        with stage_timer(
            self.config.run_dir,
            "retriever.vectorstore.init",
            request_id=request_id,
            strategy_name=strategy_name,
            filename_filter=filename_filter,
        ):
            vectorstore = self._vectorstore()
        search_filter = _metadata_filter(strategy_name=strategy_name, filename_filter=filename_filter)
        search_kwargs: dict[str, Any] = {"k": top_k}
        if search_filter:
            search_kwargs["filter"] = search_filter
        return vectorstore.as_retriever(search_kwargs=search_kwargs)

    def _retrieve_summary_candidates(
        self,
        query: str,
        strategy_name: str,
        top_k: int,
        filename_filter: str,
        request_id: str | None = None,
    ) -> list[Any]:
        with stage_timer(
            self.config.run_dir,
            "retriever.summary_candidates",
            request_id=request_id,
            strategy_name=strategy_name,
            filename_filter=filename_filter,
            top_k=top_k,
            query_len=len(query),
        ) as event:
            search_filter = _metadata_filter(strategy_name=strategy_name, filename_filter=filename_filter)
            try:
                with stage_timer(
                    self.config.run_dir,
                    "retriever.vectorstore.init",
                    request_id=request_id,
                    strategy_name=strategy_name,
                    filename_filter=filename_filter,
                ):
                    vectorstore = self._vectorstore()
                rows = vectorstore.similarity_search_with_relevance_scores(
                    query,
                    k=top_k,
                    filter=search_filter or None,
                )
                documents = [_with_summary_score(document, score) for document, score in rows]
                event["backend"] = "relevance_scores"
                event["returned_docs"] = len(documents)
                return documents
            except Exception:
                try:
                    rows = vectorstore.similarity_search_with_score(
                        query,
                        k=top_k,
                        filter=search_filter or None,
                    )
                    documents = [_with_summary_score(document, _score_from_distance(score)) for document, score in rows]
                    event["backend"] = "distance_scores"
                    event["returned_docs"] = len(documents)
                    return documents
                except Exception:
                    documents = self._native_chroma_retrieve(
                        query=query,
                        top_k=top_k,
                        strategy_name=strategy_name,
                        filename_filter=filename_filter,
                        include_scores=True,
                        request_id=request_id,
                    )
                    event["backend"] = "native_chroma_fallback"
                    event["returned_docs"] = len(documents)
                    return documents

    def _native_chroma_retrieve(
        self,
        query: str,
        top_k: int,
        strategy_name: str,
        filename_filter: str | None,
        include_scores: bool,
        request_id: str | None = None,
    ) -> list[Any]:
        with stage_timer(
            self.config.run_dir,
            "retriever.native_chroma",
            request_id=request_id,
            strategy_name=strategy_name,
            filename_filter=filename_filter,
            top_k=top_k,
            include_scores=include_scores,
            query_len=len(query),
        ) as event:
            latest = load_latest_chroma_index(self.config)
            persist_dir = resolve_chroma_persist_dir(self.config, latest)
            collection_name = resolve_collection_name(latest)
            search_filter = _metadata_filter(strategy_name=strategy_name, filename_filter=filename_filter)
            embeddings = _openai_embeddings(self.config)
            with stage_timer(
                self.config.run_dir,
                "retriever.native_chroma.embed_query",
                request_id=request_id,
                strategy_name=strategy_name,
                filename_filter=filename_filter,
                query_len=len(query),
            ):
                query_embedding = embeddings.embed_query(query)

            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("chromadb is required for native Chroma retrieval fallback.") from exc

            collection = _chroma_collection(str(persist_dir), collection_name)
            with stage_timer(
                self.config.run_dir,
                "retriever.native_chroma.collection_query",
                request_id=request_id,
                strategy_name=strategy_name,
                filename_filter=filename_filter,
                top_k=top_k,
            ):
                result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=search_filter or None,
                    include=["metadatas", "documents", "distances"],
                )
            documents = _documents_from_chroma_result(result, include_scores=include_scores)
            event["returned_docs"] = len(documents)
            return documents

    def _retrieve_summary_coverage_candidates(
        self,
        strategy_name: str,
        filename_filter: str,
        request_id: str | None = None,
    ) -> list[Any]:
        with stage_timer(
            self.config.run_dir,
            "retriever.summary_coverage_candidates",
            request_id=request_id,
            strategy_name=strategy_name,
            filename_filter=filename_filter,
        ) as event:
            latest = load_latest_chroma_index(self.config)
            run_id = str(latest.get("run_id") or "")
            persist_dir = resolve_chroma_persist_dir(self.config, latest)
            collection_name = str(resolve_collection_name(latest) or "")
            search_filter = _metadata_filter(strategy_name=strategy_name, filename_filter=filename_filter)
            coverage_key = _summary_coverage_cache_key(
                run_id=run_id,
                collection_name=collection_name,
                strategy_name=strategy_name,
                filename_filter=filename_filter,
            )
            cached_payloads = _summary_coverage_cache_get(coverage_key)
            if cached_payloads is not None:
                documents = _documents_from_cached_payloads(cached_payloads)
                event["cache_hit"] = True
                event["returned_docs"] = len(documents)
                return documents

            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("chromadb is required for summary coverage retrieval.") from exc

            collection = _chroma_collection(str(persist_dir), collection_name)
            with stage_timer(
                self.config.run_dir,
                "retriever.summary_coverage_get",
                request_id=request_id,
                strategy_name=strategy_name,
                filename_filter=filename_filter,
            ):
                result = collection.get(
                    where=search_filter or None,
                    include=["metadatas", "documents"],
                )
            documents = _documents_from_chroma_get_result(result)
            _summary_coverage_cache_put(coverage_key, _documents_to_cached_payloads(documents))
            event["cache_hit"] = False
            event["returned_docs"] = len(documents)
            return documents

    def _vectorstore(self) -> Any:
        latest = load_latest_chroma_index(self.config)
        persist_dir = resolve_chroma_persist_dir(self.config, latest)
        collection_name = resolve_collection_name(latest)
        return _cached_vectorstore(
            persist_directory=str(persist_dir),
            collection_name=collection_name,
            embedding_model=self.config.embedding_model,
            embedding_dimensions=self.config.embedding_dimensions,
            openai_api_key=self.config.openai_api_key,
        )


def _summary_coverage_cache_key(
    run_id: str,
    collection_name: str,
    strategy_name: str,
    filename_filter: str,
) -> tuple[str, str, str, str]:
    return (run_id, collection_name, strategy_name, filename_filter)


def _summary_selection_cache_key(
    run_id: str,
    collection_name: str,
    strategy_name: str,
    filename_filter: str,
    max_chunks: int,
) -> tuple[str, str, str, str, int]:
    return (run_id, collection_name, strategy_name, filename_filter, int(max_chunks))


def _summary_coverage_cache_get(key: tuple[str, str, str, str]) -> tuple[dict[str, Any], ...] | None:
    with _summary_cache_lock:
        payloads = _summary_coverage_cache.get(key)
        if payloads is None:
            return None
        _summary_coverage_cache.move_to_end(key)
        return deepcopy(payloads)


def _summary_coverage_cache_put(key: tuple[str, str, str, str], payloads: tuple[dict[str, Any], ...]) -> None:
    with _summary_cache_lock:
        _summary_coverage_cache[key] = deepcopy(payloads)
        _summary_coverage_cache.move_to_end(key)
        while len(_summary_coverage_cache) > SUMMARY_COVERAGE_CACHE_SIZE:
            _summary_coverage_cache.popitem(last=False)


def _summary_selection_cache_get(key: tuple[str, str, str, str, int]) -> tuple[str, ...] | None:
    with _summary_cache_lock:
        chunk_ids = _summary_selection_cache.get(key)
        if chunk_ids is None:
            return None
        _summary_selection_cache.move_to_end(key)
        return tuple(chunk_ids)


def _summary_selection_cache_put(key: tuple[str, str, str, str, int], chunk_ids: tuple[str, ...]) -> None:
    with _summary_cache_lock:
        _summary_selection_cache[key] = tuple(chunk_ids)
        _summary_selection_cache.move_to_end(key)
        while len(_summary_selection_cache) > SUMMARY_SELECTION_CACHE_SIZE:
            _summary_selection_cache.popitem(last=False)


def _documents_to_cached_payloads(documents: list[Any]) -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = []
    for document in documents:
        payloads.append(
            {
                "page_content": _document_text(document),
                "metadata": deepcopy(getattr(document, "metadata", {}) or {}),
            }
        )
    return tuple(payloads)


def _documents_from_cached_payloads(payloads: tuple[dict[str, Any], ...]) -> list[Any]:
    documents: list[Any] = []
    for payload in payloads:
        metadata = deepcopy(payload.get("metadata", {}) or {})
        documents.append(
            _make_document(
                page_content=str(payload.get("page_content", "") or ""),
                metadata=metadata,
            )
        )
    return documents


def _restore_selected_documents(coverage_candidates: list[Any], cached_doc_keys: tuple[str, ...]) -> list[Any]:
    by_key = {_document_key(document): document for document in coverage_candidates}
    restored: list[Any] = []
    for key in cached_doc_keys:
        document = by_key.get(key)
        if document is not None:
            restored.append(document)
    return restored


def _metadata_filter(strategy_name: str, filename_filter: str | None) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    if strategy_name:
        filters.append({"strategy_name": strategy_name})
    if filename_filter:
        filters.append({"filename": filename_filter})
    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _summary_representative_query(filename: str) -> str:
    return (
        "Find representative explanatory body chunks for a document-wide summary. "
        "Prefer broad business context, major claims, key results, risks, and conclusions. "
        "Avoid table of contents, appendix, title-only, chart-only, image-only, caption-only, or number-only chunks. "
        f"Document: {filename}"
    )


def _summary_coverage_budget(requested_top_k: int) -> int:
    return min(SUMMARY_MAX_COVERAGE_CHUNKS, max(SUMMARY_MIN_COVERAGE_CHUNKS, int(requested_top_k) * 3))


def _select_summary_coverage_documents(documents: list[Any], max_chunks: int) -> list[Any]:
    candidates = [_with_coverage_score(document) for document in documents if _document_text(document)]
    if not candidates:
        return []

    by_page: dict[int, list[tuple[Any, float]]] = {}
    for document, score in candidates:
        page = _document_page(document)
        by_page.setdefault(page, []).append((document, score))
    pages = sorted(page for page in by_page if page > 0)
    if not pages:
        pages = sorted(by_page)

    selected: list[Any] = []
    seen_ids: set[str] = set()
    page_budget = min(len(pages), max_chunks)
    coverage_pages = _choose_coverage_pages(by_page, pages, page_budget)
    for page in coverage_pages:
        document = _best_document_for_page(by_page.get(page, []), seen_ids)
        if document is not None:
            _mark_summary_selection(document, "page_coverage")
            selected.append(document)
            seen_ids.add(_document_key(document))

    remaining = max_chunks - len(selected)
    if remaining > 0:
        fill_candidates = sorted(
            candidates,
            key=lambda item: (_coverage_score(item[0]), -_document_page(item[0])),
            reverse=True,
        )
        page_counts = _page_counts(selected)
        for document, _score in fill_candidates:
            if len(selected) >= max_chunks:
                break
            key = _document_key(document)
            if key in seen_ids:
                continue
            page = _document_page(document)
            if page_counts.get(page, 0) >= 2:
                continue
            _mark_summary_selection(document, "importance_fill")
            selected.append(document)
            seen_ids.add(key)
            page_counts[page] = page_counts.get(page, 0) + 1

    return sorted(selected, key=lambda document: (_document_page(document), _document_chunk_index(document)))


def _choose_coverage_pages(
    by_page: dict[int, list[tuple[Any, float]]],
    pages: list[int],
    page_budget: int,
) -> list[int]:
    if len(pages) <= page_budget:
        return pages
    scored_pages = sorted(
        pages,
        key=lambda page: max((_coverage_score(document) for document, _ in by_page.get(page, [])), default=0.0),
        reverse=True,
    )
    selected = set(scored_pages[: max(1, page_budget // 2)])
    if page_budget > len(selected):
        step = (len(pages) - 1) / max(1, page_budget - len(selected) - 1)
        for index in range(page_budget - len(selected)):
            selected.add(pages[round(index * step)])
    return sorted(selected)[:page_budget]


def _best_document_for_page(page_items: list[tuple[Any, float]], seen_ids: set[str]) -> Any | None:
    available = [(document, score) for document, score in page_items if _document_key(document) not in seen_ids]
    if not available:
        return None
    available.sort(key=lambda item: item[1], reverse=True)
    return available[0][0]


def _with_coverage_score(document: Any) -> tuple[Any, float]:
    score = _coverage_score(document)
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, dict):
        metadata["_summary_coverage_score"] = round(score, 6)
    return document, score


def _coverage_score(document: Any) -> float:
    metadata = getattr(document, "metadata", {}) or {}
    text = _document_text(document)
    lower_text = text.lower()
    chunk_type = str(metadata.get("chunk_type") or "").lower()
    visual_type = str(metadata.get("visual_type") or "").lower()
    score = 0.4

    if chunk_type in {"page", "block_group", "native_page"}:
        score += 0.28
    elif chunk_type == "table_summary":
        score += 0.22
    elif chunk_type == "chart_summary":
        score += 0.18
    elif chunk_type == "text":
        score += 0.12

    if visual_type == "table":
        score += 0.08
    elif visual_type in {"chart", "chart_like_page"}:
        score += 0.05

    score += min(0.18, len(text) / 3200.0)
    if _has_metric_signal(text):
        score += 0.08
    if _looks_explanatory_summary_text(text):
        score += 0.08

    penalty_terms = ("contents", "table of contents", "appendix", "disclaimer", "면책", "목차", "부록", "감사합니다")
    if any(term in lower_text for term in penalty_terms):
        score -= 0.2
    if len(text) < 80:
        score -= 0.15
    if chunk_type in {"title", "image"}:
        score -= 0.2

    return score


def _has_metric_signal(text: str) -> bool:
    return bool(
        any(unit in text for unit in ("억원", "조원", "%", "bp", "ROE", "BPS", "EPS", "수수료", "순이익", "세전이익"))
    )


def _looks_explanatory_summary_text(text: str) -> bool:
    if len(text) < 160:
        return False
    tokens = text.split()
    return len(tokens) >= 8 or len(text) >= 260


def _document_text(document: Any) -> str:
    return str(getattr(document, "page_content", "") or "").strip()


def _document_page(document: Any) -> int:
    metadata = getattr(document, "metadata", {}) or {}
    try:
        return int(metadata.get("page_num", metadata.get("page", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def _document_chunk_index(document: Any) -> int:
    metadata = getattr(document, "metadata", {}) or {}
    try:
        return int(metadata.get("chunk_index") or 0)
    except (TypeError, ValueError):
        return 0


def _document_key(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    for key in ("vector_record_id", "chunk_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return f"{_document_page(document)}:{_document_chunk_index(document)}:{hash(_document_text(document))}"


def _page_counts(documents: list[Any]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for document in documents:
        page = _document_page(document)
        counts[page] = counts.get(page, 0) + 1
    return counts


def _mark_summary_selection(document: Any, reason: str) -> None:
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, dict):
        metadata["_summary_selection_reason"] = reason


def _with_summary_score(document: Any, score: float) -> Any:
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, dict):
        metadata["_summary_raw_score"] = float(score)
    return document


def _score_from_distance(distance: float) -> float:
    try:
        return 1.0 - (float(distance) / 2.0)
    except (TypeError, ValueError):
        return 0.0


def _documents_from_chroma_result(result: dict[str, Any], include_scores: bool) -> list[Any]:
    ids = _first_result_list(result.get("ids"))
    metadatas = _first_result_list(result.get("metadatas"))
    documents = _first_result_list(result.get("documents"))
    distances = _first_result_list(result.get("distances"))
    output: list[Any] = []
    for index, raw_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        metadata = dict(metadata)
        if raw_id and not metadata.get("vector_record_id"):
            metadata["vector_record_id"] = str(raw_id)
        if include_scores and index < len(distances):
            metadata["_summary_raw_score"] = _score_from_distance(distances[index])
        page_content = str(documents[index] if index < len(documents) else "")
        output.append(_make_document(page_content=page_content, metadata=metadata))
    return output


def _documents_from_chroma_get_result(result: dict[str, Any]) -> list[Any]:
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    documents = result.get("documents") or []
    output: list[Any] = []
    for index, raw_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        metadata = dict(metadata)
        if raw_id and not metadata.get("vector_record_id"):
            metadata["vector_record_id"] = str(raw_id)
        page_content = str(documents[index] if index < len(documents) else "")
        output.append(_make_document(page_content=page_content, metadata=metadata))
    return output


def _first_result_list(value: Any) -> list[Any]:
    if isinstance(value, list) and value:
        first = value[0]
        return first if isinstance(first, list) else value
    return []


def _make_document(page_content: str, metadata: dict[str, Any]) -> Any:
    try:
        from langchain_core.documents import Document

        return Document(page_content=page_content, metadata=metadata)
    except ImportError:
        return RetrievedDocument(page_content=page_content, metadata=metadata)


@dataclass
class RetrievedDocument:
    page_content: str
    metadata: dict[str, Any]


def _openai_embeddings(config: RagApiConfig) -> Any:
    try:
        return _cached_openai_embeddings(
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            api_key=config.openai_api_key,
        )
    except ImportError as exc:
        raise RuntimeError("langchain-openai is required for OpenAI embeddings.") from exc


def _chroma_class() -> Any:
    try:
        return _cached_chroma_class()
    except ImportError as exc:
        raise RuntimeError("langchain-chroma or langchain-community is required for Chroma retrieval.") from exc


@lru_cache(maxsize=8)
def _cached_openai_embeddings(model: str, dimensions: int, api_key: str) -> Any:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model,
        dimensions=dimensions,
        api_key=api_key,
    )


@lru_cache(maxsize=4)
def _cached_chroma_class() -> Any:
    try:
        from langchain_chroma import Chroma

        return Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

        return Chroma


@lru_cache(maxsize=16)
def _cached_vectorstore(
    persist_directory: str,
    collection_name: str | None,
    embedding_model: str,
    embedding_dimensions: int,
    openai_api_key: str,
) -> Any:
    chroma_cls = _chroma_class()
    embeddings = _cached_openai_embeddings(
        model=embedding_model,
        dimensions=embedding_dimensions,
        api_key=openai_api_key,
    )
    kwargs: dict[str, Any] = {
        "persist_directory": persist_directory,
        "embedding_function": embeddings,
    }
    if collection_name:
        kwargs["collection_name"] = collection_name
    return chroma_cls(**kwargs)


@lru_cache(maxsize=8)
def _persistent_chroma_client(path: str) -> Any:
    import chromadb

    return chromadb.PersistentClient(path=path)


@lru_cache(maxsize=16)
def _chroma_collection(path: str, collection_name: str | None) -> Any:
    client = _persistent_chroma_client(path)
    return client.get_collection(collection_name) if collection_name else client.list_collections()[0]
