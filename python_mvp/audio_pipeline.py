"""Audio decoding, log-mel feature extraction, and multi-segment processing.
Database-free, torchaudio-free, pure PyTorch + ffmpeg.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
SEGMENT_DURATION = 4.0  # seconds per segment
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)  # 88,200 samples


def create_mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
                          f_min: float = 20.0, f_max: float = 11025.0) -> torch.Tensor:
    """Deterministic Slaney-normalized Mel filterbank in pure PyTorch."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fftfreqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for i in range(n_mels):
        lower = (fftfreqs - hz_points[i]) / (hz_points[i + 1] - hz_points[i])
        upper = (hz_points[i + 2] - fftfreqs) / (hz_points[i + 2] - hz_points[i + 1])
        weights = np.maximum(0.0, np.minimum(lower, upper))
        # Slaney area normalization
        enorm = 2.0 / (hz_points[i + 2] - hz_points[i])
        filters[i] = weights * enorm

    return torch.from_numpy(filters)


_MEL_FB_CACHE: Optional[torch.Tensor] = None


def get_mel_filterbank() -> torch.Tensor:
    global _MEL_FB_CACHE
    if _MEL_FB_CACHE is None:
        _MEL_FB_CACHE = create_mel_filterbank()
    return _MEL_FB_CACHE


def decode_audio(audio_path: Union[str, Path], target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Decode audio file to 1D mono float32 waveform at target_sr using ffmpeg."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(path),
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg decoding failed for {path}: {e.stderr.decode('utf-8', errors='replace')}") from e

    raw = proc.stdout
    if not raw:
        raise RuntimeError(f"FFmpeg returned empty stream for {path}")

    waveform_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(waveform_np)


def compute_log_mel(waveform: torch.Tensor, sr: int = SAMPLE_RATE, n_fft: int = N_FFT,
                    hop_length: int = HOP_LENGTH, n_mels: int = N_MELS,
                    mel_fb: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute 128-bin log-mel spectrogram from 1D mono waveform."""
    if mel_fb is None:
        mel_fb = get_mel_filterbank()

    device = waveform.device
    window = torch.hann_window(n_fft, device=device)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    power_spec = stft.abs().square()
    mel_spec = torch.matmul(mel_fb.to(device), power_spec)
    log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
    return log_mel


def extract_segments(waveform: torch.Tensor, segment_samples: int = SEGMENT_SAMPLES,
                     num_segments: int = 3, is_training: bool = False,
                     rng: Optional[np.random.Generator] = None) -> List[torch.Tensor]:
    """Extract multiple temporal segments from waveform (early, mid, late or random crops)."""
    total_len = len(waveform)
    if total_len <= segment_samples:
        # Pad with repeat or zeros if shorter than segment_samples
        repeats = math.ceil(segment_samples / max(1, total_len))
        padded = waveform.repeat(repeats)[:segment_samples]
        return [padded for _ in range(num_segments)]

    max_start = total_len - segment_samples
    segments = []

    if is_training and rng is not None:
        for _ in range(num_segments):
            start = rng.integers(0, max_start + 1)
            segments.append(waveform[start:start + segment_samples])
    else:
        # Deterministic canonical positions: early (15%), mid (50%), late (85%)
        fractions = np.linspace(0.15, 0.85, num_segments)
        for frac in fractions:
            start = int(frac * max_start)
            start = max(0, min(start, max_start))
            segments.append(waveform[start:start + segment_samples])

    return segments


def extract_track_log_mels(audio_path: Union[str, Path], num_segments: int = 3,
                           is_training: bool = False,
                           rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """Decode audio and return stack of log-mel spectrograms of shape [num_segments, 1, 128, T]."""
    waveform = decode_audio(audio_path)
    segments = extract_segments(waveform, num_segments=num_segments, is_training=is_training, rng=rng)
    mel_fb = get_mel_filterbank()
    mels = [compute_log_mel(s, mel_fb=mel_fb).unsqueeze(0) for s in segments]
    return torch.stack(mels)  # [num_segments, 1, 128, T]


@dataclass
class AudioMetadata:
    track_id: str
    recording_mbid: Optional[str]
    artist_name: str
    track_title: str
    audio_path: str
    audio_sha256: str
    duration_seconds: float
    source: str
    license: str
    audio_available: bool = True

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "recording_mbid": self.recording_mbid,
            "artist_name": self.artist_name,
            "track_title": self.track_title,
            "audio_path": str(self.audio_path),
            "audio_sha256": self.audio_sha256,
            "duration_seconds": self.duration_seconds,
            "source": self.source,
            "license": self.license,
            "audio_available": self.audio_available
        }
