import os
import pandas as pd
import numpy as np

from src.feature_extraction import extract_features

DATASET_PATH = "dataset/RAVDESS"

features = []
emotions = []

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

print("\nExtracting Features...\n")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(
                root,
                file
            )

            try:

                emotion_code = file.split("-")[2]

                emotion = emotion_map[
                    emotion_code
                ]

                extracted_features = extract_features(
                    file_path
                )

                # IMPORTANT FIX
                for feature in extracted_features:

                    features.append(feature)
                    
                    emotions.append(emotion)

                print(
                    "Processed:",
                    file
                )

            except Exception as e:

                print(
                    "Error:",
                    file,
                    e
                )

# Convert to DataFrame
df = pd.DataFrame(features)

# Add label column
df["emotion"] = emotions

# Save
df.to_csv(
    "dataset/features.csv",
    index=False
)

print("\nDataset Created Successfully")

print(df.shape)