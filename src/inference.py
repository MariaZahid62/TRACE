"""
Transcribe a single .wav file using a trained checkpoint.

Usage:
    python -m src.inference --checkpoint checkpoints/best.pt --audio samples/example.wav
"""

from __future__ import annotations

import argparse

import torch
import torchaudio
import soundfile as sf

from src.dataset import FeatureExtractor, VOCAB_SIZE
from src.decoder import greedy_decode
from src.model import DeepSpeechLite
from src.utils import get_device


def load_model(checkpoint_path: str, device: torch.device) -> DeepSpeechLite:
    state = torch.load(checkpoint_path, map_location=device)
    cfg = state["cfg"]
    model = DeepSpeechLite(
        n_mels=cfg["model"]["n_mels"],
        vocab_size=VOCAB_SIZE,
        cnn_channels=cfg["model"]["cnn_channels"],
        rnn_hidden_size=cfg["model"]["rnn_hidden_size"],
        rnn_layers=cfg["model"]["rnn_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, cfg


def transcribe(model, cfg, audio_path: str, device: torch.device) -> str:
    data, sample_rate = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)
    target_sr = cfg["data"]["sample_rate"]
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # downmix to mono

    fe = FeatureExtractor(
        sample_rate=target_sr,
        n_mels=cfg["data"]["n_mels"],
        n_fft=cfg["data"]["n_fft"],
        hop_length=cfg["data"]["hop_length"],
        win_length=cfg["data"]["win_length"],
    )
    features = fe(waveform).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_mels, time)
    lengths = torch.tensor([features.shape[-1]])

    with torch.no_grad():
        log_probs, output_lengths = model(features, lengths.to(device))

    transcript = greedy_decode(log_probs.cpu(), output_lengths.cpu())[0]
    return transcript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    args = parser.parse_args()

    device = get_device()
    model, cfg = load_model(args.checkpoint, device)
    transcript = transcribe(model, cfg, args.audio, device)
    print(f"Transcript: {transcript}")


if __name__ == "__main__":
    main()
