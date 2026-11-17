"""Log-mel features: 16 kHz audio, 25 ms window, 10 ms hop, 80 bands."""
import numpy as np
import librosa

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
_WINDOW = np.hanning(N_FFT).astype(np.float32)
_MEL = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=40, fmax=7600).astype(np.float32)


def log_mel(wav: np.ndarray) -> np.ndarray:
    """Return an array of shape (frames, 80) from mono float audio at 16 kHz."""
    if len(wav) < N_FFT:
        wav = np.pad(wav, (0, N_FFT - len(wav)))
    n = 1 + (len(wav) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(wav[idx] * _WINDOW, axis=1)) ** 2
    return np.log(np.maximum(spec @ _MEL.T, 1e-10)).astype(np.float32)


def normalize(mel: np.ndarray) -> np.ndarray:
    """Zero mean and unit variance per utterance, so loudness and mic gain drop out."""
    return (mel - mel.mean(0, keepdims=True)) / (mel.std(0, keepdims=True) + 1e-5)
