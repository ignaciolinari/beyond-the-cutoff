#!/usr/bin/env python3
"""Streamlit app for human annotation of model outputs.

This app provides a UI for human evaluators to:
- Compare pairs of model outputs side-by-side
- Record preferences (A wins, B wins, or tie)
- Add optional rationale and confidence levels
- Track annotation progress
- Export annotations for ELO calculation
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

# Handle Streamlit import gracefully
try:
    import streamlit as st
except ModuleNotFoundError as err:
    raise RuntimeError(
        "Streamlit is required for apps/human_annotation.py. " "Install with: pip install streamlit"
    ) from err

# Import our evaluation modules
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beyond_the_cutoff.evaluation.elo_ranking import (
    compute_elo_rankings,
    head_to_head_matrix,
    save_leaderboard,
)
from beyond_the_cutoff.evaluation.human_evaluation import (
    AnnotationBatch,
    AnnotationStatus,
    AnnotationTask,
    HumanAnnotation,
    compute_agreement_stats,
    export_annotations_for_elo,
    load_annotation_batch,
    save_annotation_batch,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_BATCH_DIR = Path("evaluation/human_annotations")
DEFAULT_TASKS_FILE = Path("evaluation/datasets/pairwise_tasks.jsonl")

VERDICT_OPTIONS = {
    "Response A is better": "win_a",
    "Response B is better": "win_b",
    "They are roughly equal (tie)": "tie",
}

CONFIDENCE_OPTIONS = ["low", "medium", "high"]

FLAG_OPTIONS = [
    "unclear_question",
    "both_responses_wrong",
    "both_responses_correct",
    "need_domain_expertise",
    "contains_harmful_content",
]


# =============================================================================
# Session State Management
# =============================================================================


def init_session_state() -> None:
    """Initialize Streamlit session state."""
    if "annotator_id" not in st.session_state:
        st.session_state.annotator_id = ""
    if "current_batch" not in st.session_state:
        st.session_state.current_batch = None
    if "current_task_idx" not in st.session_state:
        st.session_state.current_task_idx = 0
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "all_batches" not in st.session_state:
        st.session_state.all_batches = []


def get_batch_dir() -> Path:
    """Get the batch directory from session state or default."""
    return Path(st.session_state.get("batch_dir", DEFAULT_BATCH_DIR))


# =============================================================================
# File I/O
# =============================================================================


def load_tasks_from_jsonl(path: Path) -> list[AnnotationTask]:
    """Load annotation tasks from a JSONL file."""
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                tasks.append(AnnotationTask.from_dict(data))
    return tasks


def list_existing_batches(batch_dir: Path) -> list[Path]:
    """List all existing batch files."""
    if not batch_dir.exists():
        return []
    return sorted(batch_dir.glob("*.json"))


def load_all_batches(batch_dir: Path) -> list[AnnotationBatch]:
    """Load all annotation batches from a directory."""
    batches = []
    for path in list_existing_batches(batch_dir):
        try:
            batches.append(load_annotation_batch(path))
        except Exception as e:
            st.warning(f"Failed to load batch {path}: {e}")
    return batches


# =============================================================================
# UI Components
# =============================================================================


def render_sidebar() -> str:
    """Render the sidebar with navigation and settings."""
    st.sidebar.title("Tag Human Annotation")

    # Annotator ID
    annotator_id = st.sidebar.text_input(
        "Your Annotator ID",
        value=st.session_state.annotator_id,
        placeholder="e.g., annotator_1",
    )
    if annotator_id != st.session_state.annotator_id:
        st.session_state.annotator_id = annotator_id

    st.sidebar.divider()

    # Navigation
    page: str = str(
        st.sidebar.radio(
            "Navigation",
            ["Notes Annotate", "Stats Progress", "Winner Leaderboard", "Settings"],
        )
        or "Notes Annotate"
    )

    st.sidebar.divider()

    # Quick stats
    if st.session_state.current_batch:
        batch = st.session_state.current_batch
        st.sidebar.metric(
            "Current Batch Progress",
            f"{len(batch.annotations)}/{len(batch.tasks)}",
        )

    return page


def render_annotation_page() -> None:
    """Render the main annotation interface."""
    st.title("Model Output Comparison")

    if not st.session_state.annotator_id:
        st.warning("WARNING:  Please enter your Annotator ID in the sidebar.")
        return

    # Batch selection
    batch_dir = get_batch_dir()
    col1, col2 = st.columns([3, 1])

    with col1:
        batch_options = ["Create New Batch"] + [p.stem for p in list_existing_batches(batch_dir)]
        selected_batch = st.selectbox("Select Batch", batch_options)

    with col2:
        if st.button("Load/Create"):
            if selected_batch == "Create New Batch":
                create_new_batch()
            else:
                load_batch(batch_dir / f"{selected_batch}.json")

    # Show current task
    if st.session_state.current_batch:
        render_current_task()
    else:
        st.info("Up Select or create a batch to start annotating.")


def create_new_batch() -> None:
    """Create a new annotation batch."""
    tasks_file = Path(st.session_state.get("tasks_file", DEFAULT_TASKS_FILE))

    if not tasks_file.exists():
        st.error(f"Tasks file not found: {tasks_file}")
        st.info("Create pairwise tasks first using the CLI or scripts.")
        return

    tasks = load_tasks_from_jsonl(tasks_file)

    if not tasks:
        st.error("No tasks found in the tasks file.")
        return

    # Let user select number of tasks
    n_tasks = st.number_input(
        "Number of tasks",
        min_value=1,
        max_value=len(tasks),
        value=min(20, len(tasks)),
    )

    if st.button("Create Batch"):
        import random

        selected_tasks = random.sample(tasks, n_tasks)

        batch = AnnotationBatch(
            batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            annotator_id=st.session_state.annotator_id,
            tasks=selected_tasks,
            status=AnnotationStatus.IN_PROGRESS,
        )

        batch_dir = get_batch_dir()
        batch_path = batch_dir / f"{batch.batch_id}.json"
        save_annotation_batch(batch, batch_path)

        st.session_state.current_batch = batch
        st.session_state.current_task_idx = 0
        st.success(f"Created batch with {n_tasks} tasks!")
        st.rerun()


def load_batch(path: Path) -> None:
    """Load an existing batch."""
    try:
        batch = load_annotation_batch(path)
        st.session_state.current_batch = batch
        st.session_state.current_task_idx = len(batch.annotations)
        st.success(f"Loaded batch: {batch.batch_id}")
    except Exception as e:
        st.error(f"Failed to load batch: {e}")


def render_current_task() -> None:
    """Render the current annotation task."""
    batch: AnnotationBatch = st.session_state.current_batch
    idx = st.session_state.current_task_idx

    if idx >= len(batch.tasks):
        st.success("Done All tasks in this batch are complete!")

        # Option to finalize
        if st.button("Finalize Batch"):
            batch.status = AnnotationStatus.COMPLETED
            batch.completed_at = datetime.now().isoformat()
            batch_dir = get_batch_dir()
            save_annotation_batch(batch, batch_dir / f"{batch.batch_id}.json")
            st.success("Batch finalized!")
        return

    task = batch.tasks[idx]

    # Start timer for this task
    if st.session_state.start_time is None:
        st.session_state.start_time = datetime.now()

    # Progress indicator
    st.progress(idx / len(batch.tasks))
    st.caption(f"Task {idx + 1} of {len(batch.tasks)} | Type: {task.task_type}")

    # Question
    st.subheader("Question")
    st.markdown(f"> {task.question}")

    # Reference (if available)
    if task.reference:
        with st.expander("References Reference Answer"):
            st.write(task.reference)

    # Context (if available)
    if task.contexts:
        with st.expander(f"Document Retrieved Context ({len(task.contexts)} passages)"):
            for i, ctx in enumerate(task.contexts):
                st.markdown(f"**Passage {i+1}:**")
                st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
                st.divider()

    # Side-by-side responses
    st.subheader("Responses")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Response A")
        st.markdown(
            f'<div style="background-color: #f0f2f6; padding: 15px; '
            f'border-radius: 10px; min-height: 200px;">{task.response_a}</div>',
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("### Response B")
        st.markdown(
            f'<div style="background-color: #f0f2f6; padding: 15px; '
            f'border-radius: 10px; min-height: 200px;">{task.response_b}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Annotation form
    st.subheader("Your Verdict")

    verdict = st.radio(
        "Which response is better?",
        list(VERDICT_OPTIONS.keys()),
        horizontal=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        confidence = st.select_slider(
            "Confidence",
            options=CONFIDENCE_OPTIONS,
            value="medium",
        )

    with col2:
        flags = st.multiselect(
            "Flags (optional)",
            FLAG_OPTIONS,
        )

    rationale = st.text_area(
        "Rationale (optional)",
        placeholder="Explain your choice briefly...",
    )

    # Navigation buttons
    col_prev, col_skip, col_submit = st.columns([1, 1, 2])

    with col_prev:
        if idx > 0 and st.button("← Previous"):
            st.session_state.current_task_idx -= 1
            st.session_state.start_time = None
            st.rerun()

    with col_skip:
        if st.button("Skip →"):
            st.session_state.current_task_idx += 1
            st.session_state.start_time = None
            st.rerun()

    with col_submit:
        if st.button("Submit & Next →", type="primary"):
            # Calculate duration
            duration = None
            if st.session_state.start_time:
                duration = (datetime.now() - st.session_state.start_time).total_seconds()

            # Create annotation
            from beyond_the_cutoff.evaluation.elo_ranking import Outcome

            verdict_value: Outcome = VERDICT_OPTIONS[verdict]  # type: ignore[assignment]
            annotation = HumanAnnotation(
                task_id=task.task_id,
                annotator_id=st.session_state.annotator_id,
                verdict=verdict_value,
                confidence=confidence,
                rationale=rationale if rationale else None,
                duration_seconds=duration,
                flags=flags,
            )

            batch.annotations.append(annotation)

            # Save batch
            batch_dir = get_batch_dir()
            save_annotation_batch(batch, batch_dir / f"{batch.batch_id}.json")

            # Move to next
            st.session_state.current_task_idx += 1
            st.session_state.start_time = None
            st.rerun()


def render_progress_page() -> None:
    """Render the progress tracking page."""
    st.title("Stats Annotation Progress")

    batch_dir = get_batch_dir()
    batches = load_all_batches(batch_dir)

    if not batches:
        st.info("No annotation batches found.")
        return

    # Summary statistics
    total_tasks = sum(len(b.tasks) for b in batches)
    total_annotations = sum(len(b.annotations) for b in batches)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Batches", len(batches))
    col2.metric("Total Tasks", total_tasks)
    col3.metric("Total Annotations", total_annotations)

    st.divider()

    # Per-batch breakdown
    st.subheader("Batch Details")

    for batch in batches:
        with st.expander(f"{batch.batch_id} ({batch.status.value})"):
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Annotator:** {batch.annotator_id}")
            col2.write(f"**Progress:** {len(batch.annotations)}/{len(batch.tasks)}")
            col3.write(f"**Created:** {batch.created_at[:10]}")

            if batch.annotations:
                # Verdict distribution
                verdicts = [a.verdict for a in batch.annotations]
                st.bar_chart({"Verdict": verdicts})

    st.divider()

    # Inter-annotator agreement
    st.subheader("Inter-Annotator Agreement")

    # Group annotations by task_id
    annotations_by_task: dict[str, list[HumanAnnotation]] = {}
    for batch in batches:
        for annot in batch.annotations:
            if annot.task_id not in annotations_by_task:
                annotations_by_task[annot.task_id] = []
            annotations_by_task[annot.task_id].append(annot)

    agreement_stats = compute_agreement_stats(annotations_by_task)

    if agreement_stats["n_multi_annotated"] > 0:
        st.write(f"**Tasks with multiple annotations:** {agreement_stats['n_multi_annotated']}")
        st.write(f"**Raw agreement rate:** {agreement_stats['raw_agreement']:.1%}")
        if agreement_stats["fleiss_kappa"]:
            st.write(f"**Fleiss' Kappa:** {agreement_stats['fleiss_kappa']:.3f}")
    else:
        st.info("No tasks have been annotated by multiple annotators yet.")


def render_leaderboard_page() -> None:
    """Render the ELO leaderboard page."""
    st.title("Winner Model Leaderboard")

    batch_dir = get_batch_dir()
    batches = load_all_batches(batch_dir)

    if not batches:
        st.info("No annotation data available. Complete some annotations first!")
        return

    # Build task lookup
    task_lookup = {}
    for batch in batches:
        for task in batch.tasks:
            task_lookup[task.task_id] = task

    # Export to pairwise comparisons
    comparisons = export_annotations_for_elo(batches, task_lookup)

    if not comparisons:
        st.info("No completed annotations to compute rankings.")
        return

    st.write(f"**Based on {len(comparisons)} pairwise comparisons**")

    # Compute ELO
    col1, col2 = st.columns(2)
    with col1:
        k_factor = st.slider("K-Factor", 16, 64, 32)
    with col2:
        bootstrap_samples = st.slider("Bootstrap Samples", 100, 2000, 500)

    leaderboard, metadata = compute_elo_rankings(
        comparisons,
        k_factor=k_factor,
        bootstrap_samples=bootstrap_samples,
    )

    # Display leaderboard
    st.subheader("Rankings")

    for i, rating in enumerate(leaderboard):
        col1, col2, col3, col4 = st.columns([1, 3, 2, 2])

        medal = "1st" if i == 0 else "2nd" if i == 1 else "3rd" if i == 2 else f"{i+1}."
        col1.write(medal)
        col2.write(f"**{rating.model}**")

        ci_str = ""
        if rating.confidence_lower and rating.confidence_upper:
            ci_str = f" ({rating.confidence_lower:.0f}-{rating.confidence_upper:.0f})"
        col3.write(f"Rating: **{rating.rating:.0f}**{ci_str}")
        col4.write(f"W/L/T: {rating.wins}/{rating.losses}/{rating.ties}")

    st.divider()

    # Head-to-head matrix
    st.subheader("Head-to-Head Matrix")

    h2h = head_to_head_matrix(comparisons)
    models = sorted(h2h.keys())

    if len(models) > 1:
        # Create a simple table
        st.write("Win rates (row vs column):")

        rows = []
        for m1 in models:
            row = {"Model": m1}
            for m2 in models:
                if m1 == m2:
                    row[m2] = "-"
                elif m2 in h2h.get(m1, {}):
                    stats = h2h[m1][m2]
                    total = stats["wins"] + stats["losses"] + stats["ties"]
                    if total > 0:
                        win_rate = (stats["wins"] + 0.5 * stats["ties"]) / total
                        row[m2] = f"{win_rate:.0%}"
                    else:
                        row[m2] = "-"
                else:
                    row[m2] = "-"
            rows.append(row)

        st.table(rows)

    # Export option
    st.divider()
    if st.button("Export Leaderboard to JSON"):
        output_dir = batch_dir / "results"
        output_path = output_dir / f"leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_leaderboard(leaderboard, metadata, output_path)
        st.success(f"Saved to {output_path}")


def render_settings_page() -> None:
    """Render the settings page."""
    st.title("Settings")

    # Paths
    st.subheader("Data Paths")

    batch_dir = st.text_input(
        "Annotation Batch Directory",
        value=str(get_batch_dir()),
    )
    st.session_state.batch_dir = batch_dir

    tasks_file = st.text_input(
        "Tasks File (JSONL)",
        value=str(st.session_state.get("tasks_file", DEFAULT_TASKS_FILE)),
    )
    st.session_state.tasks_file = tasks_file

    st.divider()

    # Instructions
    st.subheader("Annotation Guidelines")
    st.markdown("""
    ### How to Annotate

    1. **Read the question carefully** - Understand what is being asked.

    2. **Review both responses** - Consider:
       - Factual accuracy
       - Completeness
       - Relevance to the question
       - Clarity and coherence
       - Proper citations (if applicable)

    3. **Make your choice**:
       - **Response A is better**: A is clearly superior
       - **Response B is better**: B is clearly superior
       - **Tie**: Both responses are roughly equal in quality

    4. **Add confidence**:
       - **High**: You're very confident in your choice
       - **Medium**: Reasonable confidence
       - **Low**: Hard to decide, close call

    5. **Use flags** for special cases:
       - `unclear_question`: The question is ambiguous
       - `both_responses_wrong`: Neither response is correct
       - `both_responses_correct`: Both are equally good
       - `need_domain_expertise`: Requires specialist knowledge

    ### Tips
    - Don't spend more than 2-3 minutes per task
    - When in doubt, use "tie" with medium/low confidence
    - Add rationale for borderline cases
    """)


# =============================================================================
# Main App
# =============================================================================


def main() -> None:
    """Main app entry point."""
    st.set_page_config(
        page_title="Human Annotation Tool",
        page_icon="Tag",
        layout="wide",
    )

    init_session_state()
    page = render_sidebar()

    if page == "Notes Annotate":
        render_annotation_page()
    elif page == "Stats Progress":
        render_progress_page()
    elif page == "Winner Leaderboard":
        render_leaderboard_page()
    elif page == "Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
