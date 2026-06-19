
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from src.feature_extraction import extract_features

# ==========================
# Load Model
# ==========================

model = load_model(
    "models/emotion_model.keras"
)

encoder = joblib.load(
    "models/label_encoder.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# ==========================
# Predict Function
# ==========================

def predict_emotion(audio_file):

    # Extract features
    features = extract_features(
        audio_file
    )

    # Use original feature vector only
    feature = np.array(
        features[0]
    )

    feature = feature.reshape(
        1,
        -1
    )

    feature = pd.DataFrame(feature)

    feature = scaler.transform(
        feature
    )

    prediction = model.predict(
        feature,
        verbose=0
    )

    top3_idx = np.argsort(
        prediction[0]
    )[-3:][::-1]

    top3 = []

    for idx in top3_idx:

        top3.append(
            (
                encoder.classes_[idx],
                float(prediction[0][idx])
            )
        )

    emotion = top3[0][0]

    confidence = top3[0][1]

    print("\nPrediction Vector:")
    print(prediction)

    return emotion, confidence, top3
