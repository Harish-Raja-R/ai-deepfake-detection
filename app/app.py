"""
AI Voice Deepfake Detection - Streamlit Web Application.

Production-style interactive interface for binary AI-synthesized speech detection.
Uses the verified baseline 2D CNN (models/cnn_baseline.keras) with exact
training-equivalent audio preprocessing, 128-band Log-Mel feature extraction,
and training-derived per-bin z-score normalization.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.predict import (
    CNN_EER_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_STATS_PATH,
    STANDARD_THRESHOLD,
    SUPPORTED_AUDIO_EXTS,
    VoiceDeepfakeDetector,
)

# Set page config
st.set_page_config(
    page_title="AI Voice Deepfake Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean academic interface styling
st.markdown(
    """
    <style>
        .app-title {
            font-size: 2.3rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }
        .app-subtitle {
            font-size: 1.15rem;
            color: #475569;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }
        .verdict-card-real {
            background-color: #ECFDF5;
            border: 2px solid #059669;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            margin: 1rem 0;
        }
        .verdict-card-fake {
            background-color: #FEF2F2;
            border: 2px solid #DC2626;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            margin: 1rem 0;
        }
        .info-pill {
            background-color: #F1F5F9;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.9rem;
            color: #334155;
            display: inline-block;
            margin: 0.2rem;
            font-weight: 500;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Deepfake Detection Model...")
def load_detector_cached() -> Optional[VoiceDeepfakeDetector]:
    """Cache model loading to eliminate reload latency across audio uploads."""
    if not DEFAULT_MODEL_PATH.exists():
        return None
    if not DEFAULT_STATS_PATH.exists():
        return None
    return VoiceDeepfakeDetector(
        model_path=DEFAULT_MODEL_PATH,
        stats_path=DEFAULT_STATS_PATH,
        threshold=STANDARD_THRESHOLD,
    )


def plot_audio_waveform(waveform: np.ndarray, sr: int = 16000) -> plt.Figure:
    """Plot standardized audio waveform."""
    time_axis = np.linspace(0, len(waveform) / sr, num=len(waveform))
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    ax.plot(time_axis, waveform, color="#2563EB", lw=0.8, alpha=0.9)
    ax.set_title("Processed Waveform (16 kHz, Mono, 4.0s Normalized)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_spectrogram(mel_spec_db: np.ndarray) -> plt.Figure:
    """Plot the exact 128-band Log-Mel Spectrogram fed into the CNN."""
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    cax = ax.imshow(
        mel_spec_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    cbar = fig.colorbar(cax, ax=ax, format="%+2.0f dB", pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Input Log-Mel Spectrogram (128 Mel Bins × 126 Frames)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time Frames (STFT Hop = 512)", fontsize=9)
    ax.set_ylabel("Mel Frequency Bins", fontsize=9)
    plt.tight_layout()
    return fig


def main():
    # 1. Header
    st.markdown('<div class="app-title">AI Voice Deepfake Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">ANN/CNN-based detection of AI-generated speech</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Operating Point Selection & Settings
    st.sidebar.header("⚙️ Classification Settings")
    st.sidebar.markdown("### Decision Operating Point")

    operating_point = st.sidebar.radio(
        "Select Operating Point Threshold:",
        options=["Standard Threshold (0.50)", "EER Operating Point (0.2879)"],
        index=0,
        help="Standard threshold represents the balanced baseline decision point. EER threshold reflects the Equal Error Rate operating point determined on the held-out test partition.",
    )

    if "0.2879" in operating_point:
        current_threshold = CNN_EER_THRESHOLD
        st.sidebar.info(
            "📌 **EER Threshold Selected (0.2879):**\n\n"
            "The EER threshold is an evaluation operating point derived from the held-out "
            "evaluation procedure. It is provided for demonstration and should not be interpreted "
            "as universally optimal for other datasets."
        )
    else:
        current_threshold = STANDARD_THRESHOLD
        st.sidebar.markdown(
            "📌 **Standard Threshold Selected (0.50):**\n\n"
            "Default classification boundary where posterior probability >= 0.50 flags synthetic speech."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### 📦 Verified Model Details\n"
        "- **Model:** Custom 2D CNN\n"
        "- **Checkpoint:** `models/cnn_baseline.keras`\n"
        "- **Weights:** 110,209 parameters (~430 KB)\n"
        "- **Input:** `(128, 126, 1)` Log-Mel\n"
        "- **Test ROC-AUC:** 0.9187\n"
        "- **Test EER:** 17.32%"
    )

    # Model loader
    detector = load_detector_cached()
    if detector is None:
        st.error(
            "❌ **Model Checkpoint or Preprocessing Stats Missing:**\n\n"
            "Could not find `models/cnn_baseline.keras` or `data/metadata/feature_normalization_stats.json`.\n"
            "Please verify that the project structure is intact."
        )
        return

    # 2. Audio Upload
    st.markdown("### 📤 1. Upload Audio Clip")
    uploaded_file = st.file_uploader(
        "Upload a speech sample to analyze (Supported formats: WAV, FLAC, MP3):",
        type=["wav", "flac", "mp3"],
        help="Upload an audio file containing human or synthesized speech. Supported formats: .wav, .flac, .mp3.",
    )

    if uploaded_file is None:
        st.info("💡 **Ready for input:** Upload a `.wav`, `.flac`, or `.mp3` audio recording above to begin detection.")
        
        # Display Sample Audio Options
        sample_paths = [
            PROJECT_ROOT / "data" / "raw" / "real" / "yt_0000_part_003.flac",
            PROJECT_ROOT / "data" / "raw" / "fake" / "elevenlabs_0000_part_000.flac",
        ]
        existing_samples = [p for p in sample_paths if p.exists()]
        if existing_samples:
            st.markdown("#### Or test with a verified sample from the project dataset:")
            cols = st.columns(len(existing_samples))
            for i, p in enumerate(existing_samples):
                with cols[i]:
                    label_type = "REAL (Human Voice)" if "real" in str(p) else "FAKE (AI ElevenLabs)"
                    st.markdown(f"**{label_type}** (`{p.name}`)")
                    with open(p, "rb") as af:
                        st.audio(af.read(), format="audio/flac")
        
        # Display Static Sections when no file uploaded
        render_model_info_and_limitations()
        return

    # Process Uploaded Audio
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTS:
        st.error(f"❌ Unsupported file format '{suffix}'. Please upload a WAV, FLAC, or MP3 file.")
        return

    # Write to a secure temporary file for librosa / soundfile processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)

    try:
        # 3. Audio Information & Playback
        with st.spinner("Processing audio through standardization pipeline..."):
            metadata = detector.get_audio_metadata(tmp_path)
            waveform, mel_spec_db, input_tensor = detector.process_and_extract(tmp_path)

            # Check for corrupt / empty audio
            if len(waveform) == 0 or np.all(np.abs(waveform) < 1e-6):
                st.warning("⚠️ The uploaded audio file appears to be silent or empty. Please upload an active voice recording.")
                return

        st.markdown("---")
        st.markdown("### 📋 2. Audio Information")

        c1, c2 = st.columns(2)
        with c1:
            channels = metadata["channels"]
            channel_text = "Mono (1)" if channels == 1 else f"Stereo ({channels})"
            st.markdown("**Original Upload Information:**")
            st.markdown(f"- **Filename:** `{uploaded_file.name}`")
            st.markdown(f"- **Original Sample Rate:** `{metadata['sample_rate']:,} Hz`")
            st.markdown(f"- **Duration:** `{metadata['duration_seconds']:.2f} seconds`")
            st.markdown(f"- **Channels:** `{channel_text}`")
            st.markdown(f"- **File Format:** `{metadata['format']} ({suffix.upper()})`")

        with c2:
            st.markdown("**Standardized Model Pipeline Information:**")
            st.markdown("- **Model Target Sample Rate:** `16,000 Hz (Mono)`")
            st.markdown("- **Model Duration:** `4.00 seconds (64,000 samples)`")
            st.markdown("- **Silence Handling:** `Trimmed at 20.0 dB threshold`")
            st.markdown("- **Amplitude Normalization:** `Peak Normalized (|max| = 1.0)`")
            st.markdown("- **Feature Representation:** `128-band Log-Mel Spectrogram` (128 × 126 × 1)")

        # Audio Playback
        st.markdown("**Audio Playback:**")
        st.audio(uploaded_file.getvalue(), format=f"audio/{suffix.replace('.', '')}")

        # 4. Visualizations
        st.markdown("---")
        st.markdown("### 📈 3. Acoustic Visualizations")
        col_wave, col_spec = st.columns(2)

        with col_wave:
            fig_wave = plot_audio_waveform(waveform)
            st.pyplot(fig_wave)
            plt.close(fig_wave)

        with col_spec:
            fig_spec = plot_spectrogram(mel_spec_db)
            st.pyplot(fig_spec)
            plt.close(fig_spec)

        # 5. Prediction Result
        st.markdown("---")
        st.markdown("### 🧠 4. Deepfake Detection Result")

        with st.spinner("Executing neural inference..."):
            prediction = detector.predict_file(tmp_path, threshold=current_threshold)

        prob_fake = prediction["probability_fake"]
        prob_real = prediction["probability_real"]
        verdict = prediction["verdict"]
        confidence = prediction["confidence_score_pct"]
        is_deepfake = prediction["is_deepfake"]

        # Display Result Banner
        if is_deepfake:
            st.markdown(
                f"""
                <div class="verdict-card-fake">
                    <h2 style="color: #B91C1C; margin: 0; font-size: 1.8rem;">🚨 AI-GENERATED / DEEPFAKE</h2>
                    <p style="font-size: 1.15rem; color: #7F1D1D; margin: 0.5rem 0 0 0;">
                        Model confidence score: <b>{confidence:.1f}%</b>
                    </p>
                    <p style="font-size: 0.85rem; color: #991B1B; margin: 0.3rem 0 0 0;">
                        (Classification threshold: {current_threshold:.4f} | Posterior P(FAKE) = {prob_fake*100:.1f}%)
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-card-real">
                    <h2 style="color: #047857; margin: 0; font-size: 1.8rem;">✅ REAL / BONAFIDE</h2>
                    <p style="font-size: 1.15rem; color: #064E3B; margin: 0.5rem 0 0 0;">
                        Model confidence score: <b>{confidence:.1f}%</b>
                    </p>
                    <p style="font-size: 0.85rem; color: #065F46; margin: 0.3rem 0 0 0;">
                        (Classification threshold: {current_threshold:.4f} | Posterior P(REAL) = {prob_real*100:.1f}%)
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "⚠️ **Scientific Labeling Notice:** The score above reflects the model's posterior probability "
            "based on learned acoustic patterns and is **not** a mathematically guaranteed percentage of absolute truth."
        )

        # Probability Indicators
        st.markdown("#### Probability Score Breakdown")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.metric("REAL (Human) Probability", f"{prob_real * 100:.2f}%")
            st.progress(prob_real)
        with c_p2:
            st.metric("AI-GENERATED (Fake) Probability", f"{prob_fake * 100:.2f}%")
            st.progress(prob_fake)

    except Exception as exc:
        st.error(
            f"❌ **Audio Processing or Inference Error:**\n\n"
            f"An error occurred while analyzing the audio file: `{str(exc)}`.\n\n"
            "Please ensure the file is a valid, uncorrupted audio recording."
        )
    finally:
        # Cleanup temporary file safely
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Render persistent information panels
    render_model_info_and_limitations()


def render_model_info_and_limitations():
    """Render About the Model and Limitations sections."""
    st.markdown("---")

    # 6. Model Information Panel
    with st.expander("ℹ️ About the Model", expanded=False):
        st.markdown(
            r"""
            ### Custom 2D Convolutional Neural Network (CNN)
            This application uses the verified, lightweight 2D CNN model evaluated in Phase 7:
            
            * **Model Architecture:** Custom 2D CNN (3 Conv2D Blocks + BatchNorm + ReLU + MaxPool2D + GlobalAveragePooling2D + Dense)
            * **Parameters:** **110,209** (~430.5 KB memory footprint)
            * **Training Samples:** 1,307 utterances (50% Real / 50% Fake)
            * **Validation Samples:** 276 utterances
            * **Held-Out Test Samples:** 283 utterances (Zero data leakage across partitions)
            * **Test ROC-AUC:** **0.9187** (Area under ROC curve across all thresholds)
            * **Test EER:** **17.32%** (Equal Error Rate biometric anti-spoofing benchmark)
            * **Primary Accuracy ($\tau = 0.50$):** 63.25% (100% Precision, 26.76% Recall)
            * **Calibrated EER Accuracy ($\tau = 0.2879$):** 82.69% (82.52% Precision, 83.10% Recall)
            * **Training Dataset:** `garystafford/deepfake-audio-detection`
            
            **Detection Mechanism:**  
            The model extracts 128-band Log-Mel spectrograms from 4.0-second speech signals and detects 
            subtle acoustic artifacts characteristic of neural text-to-speech vocoders, such as high-frequency 
            phase inconsistencies, harmonic envelope discontinuities, and unnatural spectral tilt.
            """
        )

    # 7. Model Limitations
    st.markdown("### ⚠️ Limitations")
    st.warning(
        """
        **Please consider the following academic and technical limitations:**
        
        1. **Academic Mini-Project Scope:** This application is developed for academic demonstration and research purposes.
        2. **Not a Forensic System:** The model is **not** a certified forensic authentication system or legal evidentiary tool.
        3. **Distribution Dependence:** Classification reliability depends strongly on the acoustic similarity between the uploaded audio and the training distribution (`garystafford/deepfake-audio-detection`).
        4. **Unseen Synthesis Engines:** Modern and emerging voice cloning technologies (e.g., zero-shot diffusion, StyleTTS2, BigVGAN) may produce acoustic profiles not represented in the training set.
        5. **Environmental Degradation:** Compression artifacts (e.g., MP3 encoding, WhatsApp audio), background noise, microphone distortions, and reverberation can significantly alter spectral features.
        6. **Probability vs. Proof:** A high model probability is a statistical correlation with synthetic voice markers and does **not** constitute absolute proof that an audio clip is genuine or fabricated.
        7. **High-Stakes Disclaimer:** This system should **never** be used as the sole basis for high-stakes decisions, identity verification, or automated content moderation.
        """
    )


if __name__ == "__main__":
    main()
