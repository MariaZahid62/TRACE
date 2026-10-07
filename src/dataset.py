"""
LibriSpeech dataset wrapper: loads waveforms via soundfile, converts to
log-mel spectrograms, and tokenizes transcripts at the character level.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, Subset


# Character vocabulary: CTC blank at index 0, then space + a-z + apostrophe
CHARS = ["_", " ", "'"] + list("abcdefghijklmnopqrstuvwxyz")
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

BLANK_IDX = 0
VOCAB_SIZE = len(CHARS)


def text_to_indices(text: str) -> list[int]:
    text = text.lower().strip()
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


def indices_to_text(indices: list[int]) -> str:
    return "".join(IDX2CHAR[i] for i in indices)


class FeatureExtractor:
    """Waveform -> log-mel spectrogram."""

    def __init__(
        self,
        sample_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        win_length=400,
    ):
        self.sample_rate = sample_rate

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
        )

        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.mel(waveform)
        spec = self.to_db(spec)
        spec = spec.squeeze(0)

        # Per-utterance normalization
        spec = (spec - spec.mean()) / (spec.std() + 1e-5)

        return spec


class LibriSpeechASRDataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str,
        feature_extractor: FeatureExtractor,
        download: bool = True,
    ):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root,
            url=split,
            download=download,
        )

        self.dataset = self.dataset
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        base_dataset = self.dataset.dataset

        # Get the original index before Subset
        original_idx = self.dataset.indices[idx]

        # Get LibriSpeech metadata
        metadata = base_dataset.get_metadata(original_idx)

        # metadata[0] = actual relative audio path
        # Example:
        # dev-clean/1272/128104/1272-128104-0000.flac
        file_path = Path(base_dataset._archive) / metadata[0]

        # metadata[1] = sample rate
        # metadata[2] = transcript
        sample_rate = metadata[1]
        transcript = metadata[2]

        # Load audio using SoundFile
        data, loaded_sample_rate = sf.read(
            str(file_path),
            dtype="float32",
        )

        waveform = torch.from_numpy(data)

        # Convert to PyTorch format: (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        assert loaded_sample_rate == self.feature_extractor.sample_rate, (
            f"Expected {self.feature_extractor.sample_rate}Hz audio, "
            f"got {loaded_sample_rate}Hz"
        )

        # Extract log-mel spectrogram
        features = self.feature_extractor(waveform)

        # Convert transcript to character indices
        target = torch.tensor(
            text_to_indices(transcript),
            dtype=torch.long,
        )

        return features, target, transcript


def collate_fn(batch):
    """Pads variable-length spectrograms and targets for CTC training."""

    features, targets, transcripts = zip(*batch)

    input_lengths = torch.tensor(
        [f.shape[1] for f in features],
        dtype=torch.long,
    )

    target_lengths = torch.tensor(
        [len(t) for t in targets],
        dtype=torch.long,
    )

    n_mels = features[0].shape[0]
    max_time = int(input_lengths.max().item())
    batch_size = len(features)

    padded_features = torch.zeros(
        batch_size,
        1,
        n_mels,
        max_time,
    )

    for i, f in enumerate(features):
        padded_features[i, 0, :, :f.shape[1]] = f

    padded_targets = (
        torch.cat(targets)
        if targets
        else torch.tensor([], dtype=torch.long)
    )

    return (
        padded_features,
        padded_targets,
        input_lengths,
        target_lengths,
        transcripts,
    )