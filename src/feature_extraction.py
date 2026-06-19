import librosa
import numpy as np


# ==========================
# Add Noise
# ==========================

def add_noise(data):

    noise = 0.005 * np.random.randn(len(data))

    return data + noise


# ==========================
# Pitch Shift
# ==========================

def pitch_shift(data, sample_rate):

    return librosa.effects.pitch_shift(
        y=data,
        sr=sample_rate,
        n_steps=2
    )


# ==========================
# Extract Feature Vector
# ==========================

def get_features(audio, sample_rate):

    result = []

    # MFCC
    mfcc = np.mean(
        librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        ).T,
        axis=0
    )

    result.extend(mfcc)

    # Chroma
    stft = np.abs(
        librosa.stft(audio)
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(
            S=stft,
            sr=sample_rate
        ).T,
        axis=0
    )

    result.extend(chroma)

    # Mel Spectrogram
    mel = np.mean(
        librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate
        ).T,
        axis=0
    )

    result.extend(mel)

    # Spectral Contrast
    contrast = np.mean(
        librosa.feature.spectral_contrast(
            S=stft,
            sr=sample_rate
        ).T,
        axis=0
    )

    result.extend(contrast)

    return result


# ==========================
# Main Feature Function
# ==========================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=22050
    )

    all_features = []

    # Original
    original = get_features(
        audio,
        sample_rate
    )

    all_features.append(original)

    # Noise
    noise_audio = add_noise(audio)

    noise_feature = get_features(
        noise_audio,
        sample_rate
    )

    all_features.append(noise_feature)

    # Pitch Shift
    pitch_audio = pitch_shift(
        audio,
        sample_rate
    )

    pitch_feature = get_features(
        pitch_audio,
        sample_rate
    )

    all_features.append(pitch_feature)

    return all_features