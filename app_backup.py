"""
Gradio web demo: upload or record audio, get a live transcript.

Run: python app.py
"""

import gradio as gr
import torch
import torchaudio

from src.dataset import FeatureExtractor, VOCAB_SIZE
from src.decoder import greedy_decode
from src.model import DeepSpeechLite
from src.utils import get_device

CHECKPOINT_PATH = "checkpoints/best.pt"
device = get_device()


def load_model():
    try:
        state = torch.load(CHECKPOINT_PATH, map_location=device)
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
    except FileNotFoundError:
        return None, None


model, cfg = load_model()


def transcribe_audio(audio_path):
    if model is None:
        return "⚠️ No trained checkpoint found at checkpoints/best.pt. Train the model first with `python -m src.train`."
    if audio_path is None:
        return "Please record or upload an audio clip."

    waveform, sample_rate = torchaudio.load(audio_path)
    target_sr = cfg["data"]["sample_rate"]
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    fe = FeatureExtractor(
        sample_rate=target_sr,
        n_mels=cfg["data"]["n_mels"],
        n_fft=cfg["data"]["n_fft"],
        hop_length=cfg["data"]["hop_length"],
        win_length=cfg["data"]["win_length"],
    )
    features = fe(waveform).unsqueeze(0).unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[-1]])

    with torch.no_grad():
        log_probs, output_lengths = model(features, lengths.to(device))

    transcript = greedy_decode(log_probs.cpu(), output_lengths.cpu())[0]
    return transcript or "(no speech detected)"


with gr.Blocks(title="DeepSpeech-Lite") as demo:
    gr.Markdown(
        """
        # 🎙️ DeepSpeech-Lite
        A from-scratch CNN + BiGRU + CTC speech recognizer, trained on LibriSpeech.
        Record or upload a short clip of clear English speech.
        """
    )
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio input")
    transcript_output = gr.Textbox(label="Transcript", lines=3)
    submit_btn = gr.Button("Transcribe", variant="primary")
    submit_btn.click(fn=transcribe_audio, inputs=audio_input, outputs=transcript_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
