from pathlib import Path
import tempfile
import time

import cv2
import streamlit as st


st.set_page_config(
    page_title="Motion Tracking Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

        .stApp {
            background:
                radial-gradient(circle at 20% -5%, #1b2430 0%, transparent 36%),
                radial-gradient(circle at 100% 0%, #16212d 0%, transparent 30%),
                linear-gradient(160deg, #0c1117 0%, #111827 100%);
            color: #e5e7eb;
            font-family: "Manrope", sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            max-width: 1250px;
        }

        .hero-card {
            background: linear-gradient(125deg, #121a24 0%, #1a2533 100%);
            border: 1px solid #2a3748;
            border-radius: 14px;
            color: #f3f4f6;
            padding: 1rem 1.2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
            animation: enterUp 0.4s ease-out;
        }

        .hero-title {
            margin: 0;
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: 0.1px;
        }

        .hero-subtitle {
            margin-top: 0.4rem;
            margin-bottom: 0;
            opacity: 0.95;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .usage-card {
            background: #0f1722;
            border: 1px solid #243246;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-bottom: 1rem;
        }

        .usage-title {
            margin: 0 0 0.35rem 0;
            font-size: 0.9rem;
            color: #93c5fd;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }

        .usage-steps {
            margin: 0;
            padding-left: 1rem;
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1722 0%, #111b27 100%);
            border-right: 1px solid #263448;
        }

        section[data-testid="stSidebar"] * {
            color: #dbe4ee;
        }

        div[data-testid="stMetric"] {
            background: #101826;
            border: 1px solid #263448;
            border-radius: 12px;
            padding: 0.5rem 0.7rem;
        }

        div[data-testid="stMetricLabel"] {
            color: #93a7bc;
        }

        div[data-testid="stMetricValue"] {
            color: #f1f5f9;
        }

        .stAlert {
            border-radius: 12px;
            border: 1px solid #2b3a4f;
            background: #0f1722;
        }

        .stProgress > div > div > div {
            background: linear-gradient(90deg, #0ea5e9 0%, #22c55e 100%);
        }

        .stButton > button {
            border-radius: 10px;
            border: 1px solid #2f4157;
            background: linear-gradient(120deg, #0ea5e9 0%, #0284c7 100%);
            color: #f8fafc;
            font-weight: 700;
        }

        .stButton > button:hover {
            filter: brightness(1.08);
            border-color: #4e6f91;
        }

        @media (max-width: 900px) {
            .hero-title {
                font-size: 1.35rem;
            }
        }

        @keyframes enterUp {
            from {
                transform: translateY(10px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (4, 2, 0))


def to_rgb(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def get_video_library(root: Path) -> list[str]:
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    files = [str(path.name) for path in root.iterdir() if path.suffix.lower() in allowed and path.is_file()]
    return sorted(files)


def build_subtractor(method: str, history: int, var_threshold: int, detect_shadows: bool):
    if method == "KNN":
        return cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=var_threshold, detectShadows=detect_shadows)
    return cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=detect_shadows)


def enhance_mask(mask, kernel_size: int, morphology_steps: int):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morphology_steps)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=morphology_steps)
    _, cleaned = cv2.threshold(cleaned, 200, 255, cv2.THRESH_BINARY)
    return cleaned


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1 class="hero-title">Motion Tracking Studio</h1>
            <p class="hero-subtitle">
                Track moving objects in video with simple controls and live visual feedback.
            </p>
        </div>
        <div class="usage-card">
            <p class="usage-title">Quick Start</p>
            <ol class="usage-steps">
                <li>Choose a video source from the sidebar.</li>
                <li>Keep default settings for your first run.</li>
                <li>Click Start Tracking and watch detections live.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_styles()
    render_hero()

    root = Path(__file__).resolve().parent

    with st.sidebar:
        st.header("Control Center")
        st.caption("1) Select source  2) Tune if needed  3) Start tracking")

        input_mode = st.radio("Video Source", ["Upload file", "Use local sample"], index=0)
        available_samples = get_video_library(root)

        selected_path = None
        temp_file_path = None

        if input_mode == "Upload file":
            uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
                tmp.write(uploaded_file.read())
                tmp.close()
                temp_file_path = Path(tmp.name)
                selected_path = temp_file_path
                st.success(f"Loaded: {uploaded_file.name}")
        else:
            if available_samples:
                chosen = st.selectbox("Choose sample", available_samples)
                selected_path = root / chosen
            else:
                st.warning("No local sample videos were found in the project folder.")

        st.markdown("---")
        st.subheader("Detection Settings")
        method = st.selectbox("Background model", ["MOG2", "KNN"])
        min_area = st.slider("Minimum contour area", min_value=80, max_value=5000, value=450, step=10)
        history = st.slider("Model history", min_value=30, max_value=2000, value=400, step=10)
        var_threshold = st.slider("Sensitivity", min_value=8, max_value=100, value=25, help="Lower values detect more movement.")
        detect_shadows = st.toggle("Shadow detection", value=False)

        st.subheader("Mask cleanup")
        apply_morphology = st.toggle("Use morphology", value=True)
        kernel_size = st.slider("Kernel size", min_value=1, max_value=9, value=3, step=2)
        morphology_steps = st.slider("Cleanup iterations", min_value=1, max_value=3, value=1)

        st.subheader("Playback Settings")
        box_color = st.color_picker("Bounding box color", "#F15A24")
        playback_fps = st.slider("Playback speed (fps)", min_value=5, max_value=60, value=24)
        resize_percent = st.slider("Render scale", min_value=40, max_value=100, value=85, step=5)

        run_clicked = st.button("Start Tracking", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Choose your source and controls, then click Start Tracking.")
        st.stop()

    if selected_path is None:
        st.error("Please upload a video or pick a local sample.")
        st.stop()

    cap = cv2.VideoCapture(str(selected_path))
    if not cap.isOpened():
        st.error("Unable to open the selected video source.")
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        st.stop()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_fps_text = f"{source_fps:.2f}" if source_fps and source_fps > 0 else "Unknown"

    status_cols = st.columns(4)
    metric_total = status_cols[0].empty()
    metric_current = status_cols[1].empty()
    metric_model = status_cols[2].empty()
    metric_fps = status_cols[3].empty()

    view_col_1, view_col_2, view_col_3 = st.columns([1, 1, 1])
    with view_col_1:
        st.subheader("Original")
        original_slot = st.empty()
    with view_col_2:
        st.subheader("Tracked")
        tracked_slot = st.empty()
    with view_col_3:
        st.subheader("Foreground Mask")
        mask_slot = st.empty()

    status_text = st.empty()
    progress = st.progress(0)

    box_bgr = hex_to_bgr(box_color)
    subtractor = build_subtractor(method, history, var_threshold, detect_shadows)

    frames_processed = 0
    total_detections = 0
    frame_detections = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if resize_percent != 100:
                scale = resize_percent / 100.0
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            annotated = frame.copy()
            fg_mask = subtractor.apply(frame)

            if apply_morphology:
                fg_mask = enhance_mask(fg_mask, kernel_size=kernel_size, morphology_steps=morphology_steps)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            frame_detections = 0
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), box_bgr, 2)
                frame_detections += 1

            total_detections += frame_detections
            frames_processed += 1

            metric_total.metric("Total Detections", f"{total_detections}")
            metric_current.metric("Current Frame", f"{frame_detections}")
            metric_model.metric("Model", method)
            metric_fps.metric("Source FPS", source_fps_text)

            original_slot.image(to_rgb(frame), use_container_width=True)
            tracked_slot.image(to_rgb(annotated), use_container_width=True)
            mask_slot.image(fg_mask, channels="GRAY", use_container_width=True)

            if frame_count > 0:
                progress.progress(min(frames_processed / frame_count, 1.0))
                status_text.caption(f"Processed frame {frames_processed}/{frame_count}")
            else:
                status_text.caption(f"Processed frame {frames_processed}")

            time.sleep(1.0 / playback_fps)
    finally:
        cap.release()
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)

    st.success("Tracking session complete.")
    avg_det = (total_detections / frames_processed) if frames_processed else 0
    st.caption(f"Frames processed: {frames_processed} | Average detections per frame: {avg_det:.2f}")


if __name__ == "__main__":
    main()