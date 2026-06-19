import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import streamlit as st
import tempfile

from src.predict import predict_emotion

st.set_page_config(
    page_title="Emotion Recognition from Speech",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Emotion Recognition from Speech")

st.markdown(
    """
    Upload a WAV audio file and the AI model will
    predict the speaker's emotion.
    """
)

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        temp_file.write(
            uploaded_file.read()
        )

        emotion, confidence, top3 = predict_emotion(
            temp_file.name
        )

    st.success(
        f"Predicted Emotion: {emotion}"
    )

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    st.subheader("Top 3 Predictions")

    for emo, score in top3:

        st.write(
            f"**{emo}** : {score*100:.2f}%"
        )