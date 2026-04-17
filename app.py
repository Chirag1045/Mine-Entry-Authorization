"""
Optional Streamlit UI for testing the trained PPE model on uploaded images
or camera snapshots.
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from fusion_module import build_authorization_result, status_rows
from ppe_module import PPEDetector


@st.cache_resource
def load_detector(model_path: str | None) -> PPEDetector:
    """Cache the trained detector between Streamlit reruns."""
    return PPEDetector(model_path=model_path or None)


def decode_streamlit_image(uploaded_file) -> np.ndarray:
    """Decode a Streamlit upload or camera snapshot into BGR format."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unsupported image file.")
    return image


def render_result(image_bgr: np.ndarray, detector: PPEDetector, source_name: str) -> None:
    """Analyze an image and render the result widgets."""
    detection_result = detector.detect_frame(image_bgr, source_name=source_name)
    authorization_result = build_authorization_result(detection_result)
    annotated = detector.annotate_frame(image_bgr, detection_result, authorization_result)

    left_col, right_col = st.columns(2)
    with left_col:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
    with right_col:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Analyzed", use_container_width=True)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Helmet", authorization_result["helmet_status"].upper())
    metric_cols[1].metric("Vest", authorization_result["vest_status"].upper())
    metric_cols[2].metric("Shoes", authorization_result["shoes_status"].upper())
    metric_cols[3].metric("Decision", authorization_result["decision"])

    if authorization_result["decision"] == "ALLOW":
        st.success("FINAL DECISION: ALLOW")
    elif authorization_result["decision"] == "DENY":
        st.error("FINAL DECISION: DENY")
    else:
        st.warning("FINAL DECISION: REFRAME")

    st.write(authorization_result["message"])
    st.table(status_rows(authorization_result))


def main() -> None:
    """Streamlit UI entry point."""
    st.set_page_config(page_title="Mine PPE Entry Authorization", page_icon="M", layout="wide")
    st.title("Real-Time PPE Detection Based Smart Entry Authorization System")
    st.write("Train the model with `train.py`, then use this page to test image uploads or camera captures.")

    with st.sidebar:
        model_path = st.text_input("Model path", value="models/best.pt").strip()
        st.caption("Use the trained checkpoint saved by train.py.")

    detector = load_detector(model_path or None)

    uploaded_file = st.file_uploader("Upload a worker image", type=["jpg", "jpeg", "png"])
    camera_capture = st.camera_input("Or capture a camera snapshot")
    selected_source = uploaded_file or camera_capture

    if selected_source is not None:
        image = decode_streamlit_image(selected_source)
        render_result(image, detector, getattr(selected_source, "name", "camera_capture"))


if __name__ == "__main__":
    main()
