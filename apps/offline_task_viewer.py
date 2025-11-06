#!/usr/bin/env python3
"""Streamlit app for exploring offline dataset tasks alongside source papers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

Decorator = Callable[[Callable[..., Any]], Callable[..., Any]]


def _cache_data_stub(*_args: Any, **_kwargs: Any) -> Decorator:
    def _identity(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return _identity


try:  # pragma: no cover - optional UI dependency
    import streamlit as _streamlit
except ModuleNotFoundError:  # pragma: no cover

    class _StreamlitStub:
        cache_data: Callable[..., Decorator] = staticmethod(_cache_data_stub)

        def __getattr__(self, _name: str) -> Any:
            raise RuntimeError(
                "Streamlit is required for apps/offline_task_viewer.py. Install 'streamlit' to use this app."
            )

    _streamlit = _StreamlitStub()

st = cast(Any, _streamlit)

cache_data: Callable[..., Decorator]
if hasattr(st, "cache_data"):
    cache_data = cast(Callable[..., Decorator], st.cache_data)
else:  # pragma: no cover - fallback for stub
    cache_data = _cache_data_stub

DEFAULT_DATASET = Path("evaluation/datasets/cog_psych_offline_dataset.jsonl")
DEFAULT_RAW = Path("evaluation/datasets/cog_psych_offline_tasks.jsonl")


@cache_data(show_spinner=False)
def load_dataset(dataset_path: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path = Path(dataset_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = payload.get("metadata") or {}
            document = metadata.get("source_path")
            if not isinstance(document, str):
                continue
            grouped[document].append(payload)
    return grouped


@cache_data(show_spinner=False)
def load_raw(raw_path: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path = Path(raw_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            document = payload.get("document")
            if not isinstance(document, str):
                continue
            grouped[document].append(payload)
    return grouped


@cache_data(show_spinner=False)
def load_document(document_path: str) -> str:
    path = Path(document_path)
    if not path.exists():
        return "Document not found on disk."
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def main() -> None:
    st.set_page_config(page_title="Offline Task Viewer", layout="wide")
    st.title("Offline Task Viewer")
    st.caption("Browse generated tasks side-by-side with their source papers.")

    dataset_path = st.sidebar.text_input("Dataset JSONL", value=str(DEFAULT_DATASET))
    raw_path = st.sidebar.text_input("Raw tasks JSONL", value=str(DEFAULT_RAW))

    dataset_groups = load_dataset(dataset_path)
    raw_groups = load_raw(raw_path)

    documents = sorted({*dataset_groups.keys(), *raw_groups.keys()})
    if not documents:
        st.warning("No documents found. Check the input paths.")
        return

    search = st.sidebar.text_input("Search (path or instruction)", value="")
    if search:
        needle = search.lower()
        documents = [
            doc
            for doc in documents
            if needle in doc.lower()
            or any(
                needle in task.get("instruction", "").lower()
                for task in dataset_groups.get(doc, [])
            )
        ]
        if not documents:
            st.info("No matches. Clear the search to reset.")
            return

    default_index = 0
    selected_doc = cast(
        str,
        st.sidebar.selectbox(
            "Document",
            documents,
            index=min(default_index, len(documents) - 1),
            format_func=lambda doc: Path(doc).name,
        ),
    )

    col_doc, col_tasks = st.columns([3, 4])

    with col_doc:
        st.subheader("Paper preview")
        text = load_document(selected_doc)
        preview_length = st.slider("Preview length", 500, min(20000, len(text)), 4000, 500)
        st.text_area(
            label="Document text",
            value=text[:preview_length],
            height=600,
        )
        if len(text) > preview_length:
            st.caption(f"Showing first {preview_length:,} of {len(text):,} characters.")

    with col_tasks:
        st.subheader("Generated tasks")
        bundle_tasks = dataset_groups.get(selected_doc, [])
        bundle_raw = raw_groups.get(selected_doc, [])

        if bundle_raw:
            status_counts = Counter(rec.get("status", "unknown") for rec in bundle_raw)
            error_counts = Counter(rec.get("error") for rec in bundle_raw if rec.get("error"))
            status_text = ", ".join(f"{k}:{v}" for k, v in sorted(status_counts.items()))
            st.write(f"**Raw status:** {status_text}")
            if error_counts:
                error_text = ", ".join(f"{k}:{v}" for k, v in sorted(error_counts.items()))
                st.write(f"**Errors:** {error_text}")

        if not bundle_tasks:
            st.info("No curated tasks for this document.")
            return

        type_filter = st.multiselect(
            "Filter by task type",
            options=sorted({task.get("task_type", "") for task in bundle_tasks}),
            default=[],
        )

        for idx, task in enumerate(bundle_tasks):
            task_type = str(task.get("task_type") or "")
            if type_filter and task_type not in type_filter:
                continue
            instruction = str(task.get("instruction") or "")
            header = f"[{idx}] {task_type.upper()} — {instruction[:90]}"
            with st.expander(header, expanded=False):
                st.markdown(f"**Instruction**\n\n{instruction}")
                st.markdown("**Response**")
                st.write(task.get("expected_response", ""))

                rag = task.get("rag") or {}
                citations = rag.get("citations") or []
                contexts = rag.get("contexts") or []

                if citations:
                    st.markdown("**Citations**")
                    for citation in citations:
                        rendered = citation.get("rendered_context") or citation.get("excerpt", "")
                        st.caption(rendered)
                else:
                    st.caption("No citations attached.")

                if contexts:
                    st.markdown("**Retrieved contexts**")
                    for context_idx, context in enumerate(contexts, start=1):
                        st.write(f"_{context_idx}._ {context}")

                meta = task.get("metadata") or {}
                if meta:
                    st.markdown("**Metadata**")
                    st.json(meta)

        st.caption(
            "Use the filters to narrow task types or the sidebar search to locate documents quickly."
        )


if __name__ == "__main__":
    main()
