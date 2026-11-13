
import gradio as gr
import torch
import torchaudio
import soundfile as sf
from pathlib import Path

from src.dataset import FeatureExtractor, VOCAB_SIZE
from src.decoder import greedy_decode
from src.model import DeepSpeechLite
from src.utils import get_device


CHECKPOINT_PATH = "checkpoints_1500/best.pt"
DATA_DIR = Path("data/LibriSpeech/dev-clean/1272/128104")

device = get_device()


DEMO_RECORDINGS = {
    "Recording 0005": {
        "file": "1272-128104-0005.flac",
        "transcript": "IT IS OBVIOUSLY UNNECESSARY FOR US TO POINT OUT HOW LUMINOUS THESE CRITICISMS ARE HOW DELICATE IN EXPRESSION"
    },
    "Recording 0006": {
        "file": "1272-128104-0006.flac",
        "transcript": "ON THE GENERAL PRINCIPLES OF ART MISTER QUILTER WRITES WITH EQUAL LUCIDITY"
    },
    "Recording 0007": {
        "file": "1272-128104-0007.flac",
        "transcript": "PAINTING HE TELLS US IS OF A DIFFERENT QUALITY TO MATHEMATICS AND FINISH IN ART IS ADDING MORE FACT"
    },
    "Recording 0008": {
        "file": "1272-128104-0008.flac",
        "transcript": "AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN"
    },
    "Recording 0012": {
        "file": "1272-128104-0012.flac",
        "transcript": "ONLY UNFORTUNATELY HIS OWN WORK NEVER DOES GET GOOD"
    }
}


def load_model():
    if not Path(CHECKPOINT_PATH).exists():
        return None, None

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


model, cfg = load_model()


def load_demo(recording):
    if not recording:
        return None, ""

    info = DEMO_RECORDINGS[recording]
    audio_path = DATA_DIR / info["file"]

    if not audio_path.exists():
        return None, "Demo audio file not found."

    return str(audio_path), info["transcript"]


def transcribe_audio(audio_path):
    if model is None:
        return "Checkpoint not found. Please check checkpoints_1500/best.pt."

    if audio_path is None:
        return "Please upload or record audio first."

    try:
        audio_data, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True
        )

        waveform = torch.from_numpy(audio_data).transpose(0, 1)

        target_sr = cfg["data"]["sample_rate"]

        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sr
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        extractor = FeatureExtractor(
            sample_rate=target_sr,
            n_mels=cfg["data"]["n_mels"],
            n_fft=cfg["data"]["n_fft"],
            hop_length=cfg["data"]["hop_length"],
            win_length=cfg["data"]["win_length"],
        )

        features = extractor(waveform).unsqueeze(0).unsqueeze(0).to(device)

        lengths = torch.tensor([features.shape[-1]]).to(device)

        with torch.no_grad():
            log_probs, output_lengths = model(features, lengths)

        transcript = greedy_decode(
            log_probs.cpu(), output_lengths.cpu()
        )[0]

        return transcript if transcript else "(no speech detected)"

    except Exception as error:
        return f"Error processing audio: {error}"


with gr.Blocks(title="TRACE Speech Recognition") as demo:

    gr.Markdown(
        """
        # TRACE – English Speech Recognition
        Offline speech-to-text using a custom CNN + BiGRU + CTC model.
        """
    )

    with gr.Tab("Verified Demo Recordings"):

        demo_choice = gr.Dropdown(
            choices=list(DEMO_RECORDINGS.keys()),
            label="Select a recording"
        )

        load_button = gr.Button("Load Recording")

        demo_audio = gr.Audio(
            label="Demo Audio",
            type="filepath"
        )

        reference_output = gr.Textbox(
            label="Verified Reference Transcript",
            lines=3
        )

        load_button.click(
            fn=load_demo,
            inputs=demo_choice,
            outputs=[demo_audio, reference_output]
        )

    with gr.Tab("Try Your Own Audio"):

        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Upload or Record English Audio"
        )

        transcribe_button = gr.Button(
            "Transcribe",
            variant="primary"
        )

        prediction_output = gr.Textbox(
            label="Model Prediction",
            lines=4
        )

        transcribe_button.click(
            fn=transcribe_audio,
            inputs=audio_input,
            outputs=prediction_output
        )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
