
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input
)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# ==========================
# Create folders
# ==========================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "images",
    exist_ok=True
)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(
    "dataset/features.csv"
)

print("\nDataset Shape:")
print(df.shape)

# ==========================
# Features and Labels
# ==========================

X = df.drop(
    "emotion",
    axis=1
)

y = df["emotion"]

# ==========================
# Encode Labels
# ==========================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

# ==========================
# Scale Features
# ==========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# ==========================
# One Hot Encoding
# ==========================

y_categorical = to_categorical(
    y_encoded
)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ==========================
# Dense Neural Network
# ==========================

model = Sequential([

    Input(
        shape=(X_train.shape[1],)
    ),

    Dense(
        512,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.4),

    Dense(
        256,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.4),

    Dense(
        128,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        len(encoder.classes_),
        activation="softmax"
    )

])

# ==========================
# Compile Model
# ==========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# Early Stopping
# ==========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ==========================
# Train Model
# ==========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ==========================
# Evaluate
# ==========================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Accuracy:")
print(round(accuracy * 100, 2), "%")

# ==========================
# Save Model
# ==========================

model.save(
    "models/emotion_model.keras"
)

print("\nModel Saved Successfully")

# ==========================
# Predictions
# ==========================

y_pred = model.predict(
    X_test
)

y_pred_classes = np.argmax(
    y_pred,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)

# ==========================
# Classification Report
# ==========================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=encoder.classes_
    )
)

# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(
    y_true,
    y_pred_classes
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "images/confusion_matrix.png"
)

plt.close()

# ==========================
# Accuracy Graph
# ==========================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig(
    "images/training_accuracy.png"
)

plt.close()

# ==========================
# Loss Graph
# ==========================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.savefig(
    "images/training_loss.png"
)

plt.close()

print("\nGraphs Saved Successfully")
