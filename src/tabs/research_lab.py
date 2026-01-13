"""
Research Lab Tab for the Dual-Mode RAG Application.

This module provides a benchmarking interface for comparing
Text (Ground Truth) vs Voice (ASR/Whisper) pipelines using the
MINDS-14 dataset, calculating WER and embedding similarity metrics.
"""

from io import BytesIO
from typing import TypedDict

import numpy as np
import streamlit as st
from jiwer import wer
from sklearn.metrics.pairwise import cosine_similarity

from src.minds14_loader import MINDS14Loader, MINDS14Sample, get_available_languages
from src.openai_services import get_embedding, get_llm_response, transcribe_audio
from src.vector_db import search_similar_docs


class ExperimentResult(TypedDict):
    """
    Represents results from a pipeline experiment.

    Attributes:
        transcript (str): The input transcript text.
        embedding (list[float]): The generated embedding vector.
        retrieved_docs (list[str]): Documents retrieved from vector DB.
        llm_response (str): The LLM-generated response.
    """

    transcript: str
    embedding: list[float]
    retrieved_docs: list[str]
    llm_response: str


def _get_dataset_loader(language: str) -> MINDS14Loader:
    """
    Get or create a cached MINDS-14 dataset loader.

    Params:
        language (str): Language code (e.g., 'en-US').

    Returns:
        MINDS14Loader: The dataset loader instance.
    """
    cache_key: str = f"minds14_loader_{language}"

    if cache_key not in st.session_state:
        loader: MINDS14Loader = MINDS14Loader(language=language)
        st.session_state[cache_key] = loader

    return st.session_state[cache_key]


def _run_text_pipeline(text: str) -> ExperimentResult:
    """
    Run the text-based RAG pipeline (Pipeline A).

    Params:
        text (str): The input text query.

    Returns:
        ExperimentResult: The pipeline results including embedding,
            retrieved docs, and LLM response.
    """
    # Generate embedding
    embedding: list[float] = get_embedding(text)

    # Search for similar documents
    retrieved_docs: list[str] = search_similar_docs(
        query_vector=embedding,
        top_k=3,
    )

    # Generate LLM response
    context: str = "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else ""
    llm_response: str = get_llm_response(prompt=text, context=context)

    result: ExperimentResult = {
        "transcript": text,
        "embedding": embedding,
        "retrieved_docs": retrieved_docs,
        "llm_response": llm_response,
    }

    return result


def _run_voice_pipeline(audio_buffer: BytesIO) -> ExperimentResult:
    """
    Run the voice-based RAG pipeline (Pipeline B).

    Params:
        audio_buffer (BytesIO): Audio data as BytesIO buffer.

    Returns:
        ExperimentResult: The pipeline results including transcription,
            embedding, retrieved docs, and LLM response.
    """
    # Transcribe audio using Whisper
    transcript: str = transcribe_audio(audio_buffer)

    # Generate embedding
    embedding: list[float] = get_embedding(transcript)

    # Search for similar documents
    retrieved_docs: list[str] = search_similar_docs(
        query_vector=embedding,
        top_k=3,
    )

    # Generate LLM response
    context: str = "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else ""
    llm_response: str = get_llm_response(prompt=transcript, context=context)

    result: ExperimentResult = {
        "transcript": transcript,
        "embedding": embedding,
        "retrieved_docs": retrieved_docs,
        "llm_response": llm_response,
    }

    return result


def _calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate between reference and hypothesis text.

    Params:
        reference (str): The ground truth reference text.
        hypothesis (str): The ASR-generated hypothesis text.

    Returns:
        float: The Word Error Rate (0.0 = perfect, 1.0 = completely wrong).
    """
    if not reference.strip() or not hypothesis.strip():
        return 1.0

    error_rate: float = wer(reference.lower(), hypothesis.lower())
    return error_rate


def _calculate_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Calculate cosine similarity between two embedding vectors.

    Params:
        vec_a (list[float]): First embedding vector.
        vec_b (list[float]): Second embedding vector.

    Returns:
        float: Cosine similarity score (0.0 to 1.0).
    """
    arr_a: np.ndarray = np.array(vec_a).reshape(1, -1)
    arr_b: np.ndarray = np.array(vec_b).reshape(1, -1)

    similarity: float = float(cosine_similarity(arr_a, arr_b)[0][0])
    return similarity


def _display_pipeline_results(result: ExperimentResult, title: str) -> None:
    """
    Display results from a pipeline experiment.

    Params:
        result (ExperimentResult): The experiment results to display.
        title (str): The title for this results section.

    Returns:
        None
    """
    st.subheader(title)

    st.markdown("**Input Transcript:**")
    st.info(result["transcript"])

    st.markdown("**Retrieved Documents:**")
    for i, doc in enumerate(result["retrieved_docs"], 1):
        with st.expander(f"Document {i}"):
            st.write(doc)

    st.markdown("**LLM Response:**")
    st.success(result["llm_response"])




def render_batch_evaluation_tab() -> None:
    """
    Render the batch evaluation section for multi-accent analysis.

    Allows running batch evaluations across en-US, en-GB, en-AU locales
    and displays aggregated metrics with Plotly visualizations.

    Params:
        None

    Returns:
        None
    """
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    from src.evaluation_engine import (
        run_batch_evaluation,
        summarize_results,
        summarize_by_intent,
        INTENT_KEYWORDS,
    )

    st.header("📊 Batch Evaluation - Multi-Accent Analysis")
    st.markdown(
        "Run batch evaluations across **en-US**, **en-GB**, and **en-AU** locales "
        "to compare ASR performance and retrieval accuracy across English accents."
    )

    st.markdown("---")

    # Dataset size info
    LOCALE_DATASET_SIZES: dict[str, int] = {
        "en-US": 563,
        "en-GB": 592,
        "en-AU": 654,
    }

    # Configuration
    col_config1, col_config2 = st.columns(2)

    with col_config1:
        # Locale selection
        available_locales: list[str] = ["en-US", "en-GB", "en-AU"]
        selected_locales: list[str] = st.multiselect(
            "🌍 Select Locales",
            options=available_locales,
            default=available_locales,
            key="batch_locales",
        )

        # Show dataset sizes
        if selected_locales:
            st.caption("📊 Dataset sizes:")
            for loc in selected_locales:
                st.caption(f"  • {loc}: {LOCALE_DATASET_SIZES[loc]} samples")

    with col_config2:
        # Option to use full dataset
        use_full_dataset: bool = st.checkbox(
            "📦 Use Full Dataset (All Samples)",
            value=False,
            key="use_full_dataset",
            help="Use all available samples for each locale (en-AU: 654, en-GB: 592, en-US: 563)",
        )

        if use_full_dataset:
            samples_per_locale: int = -1  # Flag for full dataset
            total_samples_info: int = sum(LOCALE_DATASET_SIZES[loc] for loc in selected_locales)
            st.info(f"📊 Will evaluate **{total_samples_info}** total samples")
        else:
            # Samples per locale slider
            samples_per_locale = st.slider(
                "📁 Samples per Locale",
                min_value=5,
                max_value=100,
                value=10,
                step=5,
                key="batch_samples",
            )

    # Show intent keywords info
    with st.expander("📋 Intent Keywords Mapping"):
        st.markdown("Keywords used to check retrieval relevance:")
        for intent, keywords in INTENT_KEYWORDS.items():
            st.markdown(f"**{intent}**: {', '.join(keywords[:5])}...")

    st.markdown("---")

    # Run batch evaluation button
    if st.button(
        "🚀 Start Multi-Region Analysis",
        type="primary",
        use_container_width=True,
        disabled=len(selected_locales) == 0,
    ):
        if not selected_locales:
            st.error("Please select at least one locale.")
            return

        # Calculate total samples
        if use_full_dataset:
            total_samples: int = sum(LOCALE_DATASET_SIZES[loc] for loc in selected_locales)
        else:
            total_samples = len(selected_locales) * samples_per_locale
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current: int, total: int, locale: str, sample_id: str) -> None:
            """Update progress bar and status text."""
            progress: float = current / total
            progress_bar.progress(progress)
            status_text.text(f"Processing {locale} - {sample_id} ({current}/{total})")

        with st.spinner(f"Running batch evaluation on {total_samples} samples..."):
            try:
                # Run batch evaluation
                df = run_batch_evaluation(
                    locales=selected_locales,
                    samples_per_locale=samples_per_locale,
                    progress_callback=update_progress,
                )

                progress_bar.progress(1.0)
                status_text.text("✅ Evaluation complete!")

                # Store results
                st.session_state.batch_results = df
                st.session_state.batch_locales_used = selected_locales

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                return

        st.success(f"✅ Evaluated {len(df)} samples across {len(selected_locales)} locales!")
        _display_batch_results(df, selected_locales)

    # Show previous results if available
    elif "batch_results" in st.session_state:
        df = st.session_state.batch_results
        locales_used = st.session_state.get("batch_locales_used", ["en-US", "en-GB", "en-AU"])
        st.info(f"📊 Showing previous batch results ({len(df)} samples)")
        _display_batch_results(df, locales_used)


def _display_batch_results(df, locales: list[str]) -> None:
    """
    Display batch evaluation results with Plotly visualizations.

    Params:
        df: DataFrame with evaluation results.
        locales (list[str]): List of locales evaluated.

    Returns:
        None
    """
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    from src.evaluation_engine import summarize_results, summarize_by_intent

    # Get summary data
    summary_locale = summarize_results(df)

    st.markdown("---")
    st.subheader("📈 Comparative Analysis Results")

    # ===== SECTION 1: Metrics Cards =====
    st.markdown("### 📊 Key Metrics by Locale")
    metric_cols = st.columns(len(locales))
    for i, (_, row) in enumerate(summary_locale.iterrows()):
        with metric_cols[i]:
            st.markdown(f"#### 🌍 {row['locale']}")
            st.metric("Avg WER", f"{row['avg_wer_%']:.1f}%", 
                     help="Word Error Rate - Lower is better")
            st.metric("Text Hit Rate", f"{row['text_hit_rate_%']:.1f}%",
                     help="Retrieval accuracy with ground truth text")
            st.metric("Voice Hit Rate", f"{row['voice_hit_rate_%']:.1f}%",
                     help="Retrieval accuracy with ASR transcription")

    st.markdown("---")

    # ===== SECTION 2: WER Comparison Bar Chart =====
    st.markdown("### 📉 Average Word Error Rate (WER) by Locale")
    st.markdown("*Lower WER indicates better ASR accuracy for that accent.*")

    fig_wer = px.bar(
        summary_locale,
        x="locale",
        y="avg_wer_%",
        color="locale",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"locale": "Locale", "avg_wer_%": "Average WER (%)"},
        title="Word Error Rate Comparison Across English Accents",
    )
    fig_wer.update_layout(
        xaxis_title="Locale",
        yaxis_title="Average WER (%)",
        showlegend=False,
        height=400,
    )
    fig_wer.update_traces(
        texttemplate='%{y:.1f}%',
        textposition='outside',
    )
    st.plotly_chart(fig_wer, use_container_width=True)

    st.markdown("---")

    # ===== SECTION 3: Grouped Bar Chart - Text vs Voice Hit Rate =====
    st.markdown("### 🎯 Retrieval Hit Rate: Text Baseline vs Voice Experiment")
    st.markdown(
        "*This shows how ASR errors impact document retrieval accuracy. "
        "A large gap indicates ASR errors significantly degrade retrieval.*"
    )

    # Prepare data for grouped bar chart
    hit_rate_data = []
    for _, row in summary_locale.iterrows():
        hit_rate_data.append({
            "Locale": row["locale"],
            "Pipeline": "Text Baseline",
            "Hit Rate (%)": row["text_hit_rate_%"],
        })
        hit_rate_data.append({
            "Locale": row["locale"],
            "Pipeline": "Voice (ASR)",
            "Hit Rate (%)": row["voice_hit_rate_%"],
        })

    hit_rate_df = pd.DataFrame(hit_rate_data)

    fig_hit = px.bar(
        hit_rate_df,
        x="Locale",
        y="Hit Rate (%)",
        color="Pipeline",
        barmode="group",
        color_discrete_map={
            "Text Baseline": "#2ecc71",  # Green
            "Voice (ASR)": "#e74c3c",    # Red
        },
        title="Text vs Voice Retrieval Hit Rate by Locale",
    )
    fig_hit.update_layout(
        xaxis_title="Locale",
        yaxis_title="Hit Rate (%)",
        legend_title="Pipeline",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig_hit.update_traces(
        texttemplate='%{y:.1f}%',
        textposition='outside',
    )
    st.plotly_chart(fig_hit, use_container_width=True)

    # Calculate and display degradation
    st.markdown("#### 📉 ASR Impact on Retrieval (Degradation)")
    deg_cols = st.columns(len(locales))
    for i, (_, row) in enumerate(summary_locale.iterrows()):
        degradation = row["text_hit_rate_%"] - row["voice_hit_rate_%"]
        with deg_cols[i]:
            delta_color = "inverse" if degradation > 0 else "normal"
            st.metric(
                label=row["locale"],
                value=f"{row['voice_hit_rate_%']:.1f}%",
                delta=f"-{degradation:.1f}% from text",
                delta_color=delta_color,
            )

    st.markdown("---")

    # ===== SECTION 4: Summary by Intent =====
    st.markdown("### 📋 Results by Intent Class")
    summary_intent = summarize_by_intent(df)

    # Intent bar chart
    fig_intent = px.bar(
        summary_intent,
        x="intent_name",
        y=["text_hit_rate_%", "voice_hit_rate_%"],
        barmode="group",
        labels={
            "intent_name": "Intent",
            "value": "Hit Rate (%)",
            "variable": "Pipeline"
        },
        title="Hit Rate by Intent Class",
        color_discrete_map={
            "text_hit_rate_%": "#2ecc71",
            "voice_hit_rate_%": "#e74c3c",
        },
    )
    fig_intent.update_layout(
        xaxis_title="Intent",
        yaxis_title="Hit Rate (%)",
        height=500,
        xaxis_tickangle=-45,
        legend_title="Pipeline",
    )
    # Rename legend labels
    fig_intent.for_each_trace(lambda t: t.update(
        name=t.name.replace("text_hit_rate_%", "Text Baseline").replace("voice_hit_rate_%", "Voice (ASR)")
    ))
    st.plotly_chart(fig_intent, use_container_width=True)

    # Summary tables
    with st.expander("📊 Summary Tables"):
        st.markdown("**By Locale:**")
        st.dataframe(summary_locale, use_container_width=True)
        
        st.markdown("**By Intent:**")
        st.dataframe(summary_intent, use_container_width=True)

    st.markdown("---")

    # ===== SECTION 5: Raw Data & Export =====
    with st.expander("📋 Raw Evaluation Data"):
        st.dataframe(df, use_container_width=True)

    # Download button
    csv_data: str = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv_data,
        file_name="batch_evaluation_results.csv",
        mime="text/csv",
    )


def render_research_tab() -> None:
    """
    Render the Research Lab tab with sub-tabs for different analyses.

    Provides:
    - Single Sample Analysis: Compare text vs voice for one sample
    - Batch Evaluation: Multi-accent analysis across locales

    Params:
        None

    Returns:
        None
    """
    st.header("🔬 Research Lab")

    # Create sub-tabs
    tab_single, tab_batch = st.tabs([
        "🔍 Single Sample Analysis",
        "📊 Batch Evaluation",
    ])

    with tab_single:
        _render_single_sample_analysis()

    with tab_batch:
        render_batch_evaluation_tab()


def _render_single_sample_analysis() -> None:
    """
    Render the single sample analysis section.

    Provides a side-by-side comparison of Text vs Voice pipelines
    using the MINDS-14 dataset with WER and embedding similarity metrics.

    Params:
        None

    Returns:
        None
    """
    st.subheader("🔍 Single Sample Analysis")
    st.markdown(
        "Compare **Text Baseline** (Ground Truth) vs **Voice Experiment** (ASR/Whisper) "
        "pipelines using the **MINDS-14** dataset."
    )

    st.markdown("---")

    # Language selection
    col_lang, col_info = st.columns([1, 2])

    with col_lang:
        available_languages: list[str] = get_available_languages()
        selected_language: str = st.selectbox(
            "🌍 Select Language",
            options=available_languages,
            index=available_languages.index("en-US"),
            key="research_language_select",
        )

    with col_info:
        st.info(
            f"**Dataset:** [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14) | "
            f"**Language:** {selected_language}"
        )

    # Load dataset
    try:
        with st.spinner(f"Loading MINDS-14 dataset ({selected_language})..."):
            loader: MINDS14Loader = _get_dataset_loader(selected_language)
            dataset_size: int = len(loader)

        st.success(f"✅ Loaded {dataset_size} samples from MINDS-14 ({selected_language})")

    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.info("Please ensure you have installed the `datasets` library: `pip install datasets`")
        return

    # Sample selection
    sample_index: int = st.slider(
        "📁 Select Sample Index",
        min_value=0,
        max_value=dataset_size - 1,
        value=0,
        key="research_sample_index",
    )

    # Get sample
    sample: MINDS14Sample = loader.get_sample(sample_index)

    # Display sample info
    st.markdown("### 📋 Sample Information")
    col_trans, col_meta = st.columns([2, 1])

    with col_trans:
        st.markdown("**Ground Truth Transcript:**")
        st.info(sample["transcription"])

        if sample["english_transcription"]:
            st.markdown("**English Translation:**")
            st.caption(sample["english_transcription"])

    with col_meta:
        st.metric("Intent", sample["intent_name"].upper())
        st.metric("Sample ID", sample["id"])
        st.metric("Sampling Rate", f"{sample['sampling_rate']} Hz")

    # Audio player
    st.audio(sample["audio_path"], format="audio/wav")

    st.markdown("---")

    # Run experiment button
    run_experiment: bool = st.button(
        "🚀 Run Experiment",
        type="primary",
        use_container_width=True,
        key="run_single_experiment",
    )

    if run_experiment:
        ground_truth: str = sample["transcription"]

        # Use English transcription if available and language is not English
        if sample["english_transcription"] and not selected_language.startswith("en"):
            ground_truth_for_rag: str = sample["english_transcription"]
            st.info("Using English translation for RAG (better retrieval from English knowledge base)")
        else:
            ground_truth_for_rag = ground_truth

        with st.spinner("Running experiments..."):
            # Run Text Pipeline (Pipeline A)
            st.toast("Running Text Pipeline (Ground Truth)...")
            text_result: ExperimentResult = _run_text_pipeline(ground_truth_for_rag)

            # Run Voice Pipeline (Pipeline B)
            st.toast("Running Voice Pipeline (Whisper)...")
            audio_buffer: BytesIO = loader.get_audio_bytes(sample_index)
            voice_result: ExperimentResult = _run_voice_pipeline(audio_buffer)

        # Calculate metrics
        wer_score: float = _calculate_wer(
            reference=ground_truth,
            hypothesis=voice_result["transcript"],
        )
        similarity_score: float = _calculate_cosine_similarity(
            vec_a=text_result["embedding"],
            vec_b=voice_result["embedding"],
        )

        st.markdown("---")

        # Display metrics prominently
        st.subheader("📊 Experiment Metrics")
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                label="Word Error Rate (WER)",
                value=f"{wer_score:.2%}",
                help="Lower is better. 0% = perfect transcription.",
            )

        with metric_col2:
            st.metric(
                label="Embedding Similarity",
                value=f"{similarity_score:.4f}",
                help="Higher is better. 1.0 = identical embeddings.",
            )

        with metric_col3:
            # Calculate accuracy as inverse of WER
            accuracy: float = max(0.0, 1.0 - wer_score)
            st.metric(
                label="Transcription Accuracy",
                value=f"{accuracy:.2%}",
                help="Higher is better. 100% = perfect transcription.",
            )

        st.markdown("---")

        # Side-by-side comparison
        col_text, col_voice = st.columns(2)

        with col_text:
            _display_pipeline_results(
                result=text_result,
                title="📝 Pipeline A: Text Baseline",
            )

        with col_voice:
            _display_pipeline_results(
                result=voice_result,
                title="🎤 Pipeline B: Voice Experiment",
            )

        # Store results in session state for further analysis
        st.session_state.last_experiment = {
            "sample": sample,
            "language": selected_language,
            "ground_truth": ground_truth,
            "text_result": text_result,
            "voice_result": voice_result,
            "wer": wer_score,
            "similarity": similarity_score,
        }
