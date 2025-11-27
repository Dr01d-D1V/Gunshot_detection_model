#!/usr/bin/env python3
"""
Test a trained .keras gunshot detection model on audio files.

Usage:
    python src/test_model.py [--model PATH] [--threshold 0.5] [audio_dir_or_file]

Defaults:
    model      → most recent .keras in models/
    threshold  → 0.5 (sigmoid output above this → gunshot)
    audio      → data/test_samples/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from datetime import datetime

# ---------------------------------------------------------------------------
# Preprocessing (must match training pipeline in feature_extraction.ipynb)
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16000
TARGET_NUM_SAMPLES = TARGET_SAMPLE_RATE * 3  # 3 seconds
FRAME_STEP = 32
N_FFT = 512
TARGET_TIME = 1491
TARGET_FREQ = 257


def preprocess_audio(path: Path) -> np.ndarray:
    """Load audio, convert to spectrogram matching the training shape."""
    wav, _ = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    wav = wav[:TARGET_NUM_SAMPLES]
    if wav.shape[0] < TARGET_NUM_SAMPLES:
        wav = np.pad(wav, (0, TARGET_NUM_SAMPLES - wav.shape[0]), mode="constant")

    stft = librosa.stft(wav, n_fft=N_FFT, hop_length=FRAME_STEP, center=True)
    spec = np.abs(stft).T  # shape: (time, freq)

    # Pad / crop to target shape
    if spec.shape[0] > TARGET_TIME:
        spec = spec[:TARGET_TIME, :]
    elif spec.shape[0] < TARGET_TIME:
        pad_rows = TARGET_TIME - spec.shape[0]
        spec = np.pad(spec, ((0, pad_rows), (0, 0)), mode="constant")

    if spec.shape[1] > TARGET_FREQ:
        spec = spec[:, :TARGET_FREQ]
    elif spec.shape[1] < TARGET_FREQ:
        pad_cols = TARGET_FREQ - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_cols)), mode="constant")

    spec = spec.astype(np.float32)
    spec = np.expand_dims(spec, axis=-1)  # (time, freq, 1)
    return spec


def sliding_window_inference(
    audio_path: Path,
    model: tf.keras.Model,
    threshold: float,
    window_sec: float = 3.0,
    hop_sec: float = 1.0,
) -> list[dict]:
    """
    Slide a window over the audio and run inference on each chunk.
    Returns a list of detections with start/end times and confidence.
    """
    wav, sr = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True)
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    detections: list[dict] = []
    start = 0
    while start + window_samples <= len(wav):
        chunk = wav[start : start + window_samples]
        spec = preprocess_chunk(chunk)
        prob = float(model.predict(spec[np.newaxis, ...], verbose=0)[0, 0])
        if prob >= threshold:
            detections.append(
                {
                    "start_sec": start / sr,
                    "end_sec": (start + window_samples) / sr,
                    "confidence": prob,
                }
            )
        start += hop_samples

    # Handle tail if remaining audio >= half window
    remaining = len(wav) - start
    if remaining >= window_samples // 2:
        chunk = wav[start:]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)), mode="constant")
        spec = preprocess_chunk(chunk)
        prob = float(model.predict(spec[np.newaxis, ...], verbose=0)[0, 0])
        if prob >= threshold:
            detections.append(
                {
                    "start_sec": start / sr,
                    "end_sec": len(wav) / sr,
                    "confidence": prob,
                }
            )

    return detections


def preprocess_chunk(chunk: np.ndarray) -> np.ndarray:
    """Preprocess a raw waveform chunk (already at correct sample rate)."""
    if chunk.shape[0] < TARGET_NUM_SAMPLES:
        chunk = np.pad(chunk, (0, TARGET_NUM_SAMPLES - chunk.shape[0]), mode="constant")
    chunk = chunk[:TARGET_NUM_SAMPLES]

    stft = librosa.stft(chunk, n_fft=N_FFT, hop_length=FRAME_STEP, center=True)
    spec = np.abs(stft).T

    if spec.shape[0] > TARGET_TIME:
        spec = spec[:TARGET_TIME, :]
    elif spec.shape[0] < TARGET_TIME:
        spec = np.pad(spec, ((0, TARGET_TIME - spec.shape[0]), (0, 0)), mode="constant")

    if spec.shape[1] > TARGET_FREQ:
        spec = spec[:, :TARGET_FREQ]
    elif spec.shape[1] < TARGET_FREQ:
        spec = np.pad(spec, ((0, 0), (0, TARGET_FREQ - spec.shape[1])), mode="constant")

    return spec.astype(np.float32)[..., np.newaxis]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def find_latest_model(models_dir: Path) -> Path:
    keras_files = sorted(models_dir.glob("*.keras"), key=lambda p: p.stat().st_mtime)
    if not keras_files:
        raise FileNotFoundError(f"No .keras models found in {models_dir}")
    return keras_files[-1]


def collect_audio_files(path: Path) -> list[Path]:
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    default_model_dir = project_root / "models"
    default_test_dir = project_root / "data" / "test_samples"
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    parser = argparse.ArgumentParser(description="Test gunshot detection model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to .keras model (default: latest in models/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (0-1)",
    )
    parser.add_argument(
        "audio",
        type=Path,
        nargs="?",
        default=default_test_dir,
        help="Audio file or directory to test",
    )
    args = parser.parse_args()

    log_file = log_path.open("w", encoding="utf-8")

    def log_print(message: str) -> None:
        print(message)
        log_file.write(message + "\n")
        

    try:
        model_path = args.model or find_latest_model(default_model_dir)
        # print(f"Loading model: {model_path}")
        log_print(f"Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)

        audio_files = collect_audio_files(args.audio)
        if not audio_files:
            print(f"No audio files found in {args.audio}")
            sys.exit(1)

    # print(f"Testing {len(audio_files)} audio file(s) with threshold={args.threshold}\n")
        log_print(f"Testing {len(audio_files)} audio file(s) with threshold={args.threshold}\n")

        total_detections = 0
        for audio_path in audio_files:
            log_print(f"─── {audio_path.name} ───")
            detections = sliding_window_inference(audio_path, model, args.threshold)
            if detections:
                for det in detections:
                    duration = det["end_sec"] - det["start_sec"]
                    if det["start_sec"] > 60:
                        message = (
                            f"  Gunshot detected: At {(det['start_sec'] / 60):.2f} minutes "
                            f"Lasted {duration:.1f} seconds"
                            f"(confidence: {det['confidence']:.2%})"
                        )
                    else:
                        message = (
                            f"  Gunshot detected: At {det['start_sec']:.1f} seconds "
                            f"Lasted {duration:.1f} seconds "
                            f"(confidence: {det['confidence']:.2%})"
                        )
                    log_print(message)
                total_detections += len(detections)
            else:
                log_print("  No gunshots detected.")
            log_print("")
        
        log_print(f"Testing complete. \nTotal gunshot detections: {total_detections}")
        log_print(f"Log saved to: {log_path}")
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
