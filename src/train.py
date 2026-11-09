
"""
Training entry point for DeepSpeech-Lite.

Usage:
    python -m src.train --config configs/config.yaml
    python -m src.train --config configs/config.yaml --eval-only --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset import (
    LibriSpeechASRDataset,
    FeatureExtractor,
    VOCAB_SIZE,
    collate_fn,
)
from src.decoder import greedy_decode, batch_wer_cer
from src.model import DeepSpeechLite
from src.utils import set_seed, AverageMeter, save_history, get_device


def build_dataloaders(cfg: dict):
    fe = FeatureExtractor(
        sample_rate=cfg["data"]["sample_rate"],
        n_mels=cfg["data"]["n_mels"],
        n_fft=cfg["data"]["n_fft"],
        hop_length=cfg["data"]["hop_length"],
        win_length=cfg["data"]["win_length"],
    )

    train_ds = LibriSpeechASRDataset(
        cfg["data"]["root"], cfg["data"]["train_split"], fe
    )
    dev_ds = LibriSpeechASRDataset(
        cfg["data"]["root"], cfg["data"]["dev_split"], fe
    )

    # Limit dataset size for small training experiments
    train_samples = cfg["training"].get("train_samples")
    dev_samples = cfg["training"].get("dev_samples")

    if train_samples is not None:
        train_ds = Subset(
            train_ds, range(min(train_samples, len(train_ds)))
        )

    if dev_samples is not None:
        dev_ds = Subset(
            dev_ds, range(min(dev_samples, len(dev_ds)))
        )

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(dev_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
    )

    return train_loader, dev_loader


def evaluate(model, dev_loader, device, criterion):
    model.eval()
    loss_meter = AverageMeter()
    all_hyps, all_refs = [], []

    with torch.no_grad():
        for features, targets, input_lengths, target_lengths, transcripts in tqdm(
            dev_loader, desc="eval", leave=False
        ):
            features = features.to(device)
            targets = targets.to(device)

            log_probs, output_lengths = model(
                features, input_lengths.to(device)
            )
            loss = criterion(
                log_probs, targets, output_lengths.cpu(), target_lengths
            )
            loss_meter.update(loss.item(), n=features.size(0))

            hyps = greedy_decode(log_probs.cpu(), output_lengths.cpu())
            all_hyps.extend(hyps)
            all_refs.extend([t.lower() for t in transcripts])

    wer, cer = batch_wer_cer(all_hyps, all_refs)
    return loss_meter.avg, wer, cer


def train(cfg: dict, eval_only: bool = False, checkpoint: str | None = None):
    set_seed(cfg["training"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    train_loader, dev_loader = build_dataloaders(cfg)

    model = DeepSpeechLite(
        n_mels=cfg["model"]["n_mels"],
        vocab_size=VOCAB_SIZE,
        cnn_channels=cfg["model"]["cnn_channels"],
        rnn_hidden_size=cfg["model"]["rnn_hidden_size"],
        rnn_layers=cfg["model"]["rnn_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    if eval_only:
        loss, wer, cer = evaluate(model, dev_loader, device, criterion)
        print(f"[eval-only] loss={loss:.4f} WER={wer:.3f} CER={cer:.3f}")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["scheduler_step"],
        gamma=cfg["training"]["scheduler_gamma"],
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg["training"]["amp"] and device.type == "cuda"
    )

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "dev_loss": [], "wer": [], "cer": []}
    best_wer = float("inf")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{cfg['training']['epochs']}",
        )

        for step, (
            features, targets, input_lengths, target_lengths, _
        ) in enumerate(pbar):

            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=cfg["training"]["amp"] and device.type == "cuda"
            ):
                log_probs, output_lengths = model(
                    features, input_lengths.to(device)
                )
                loss = criterion(
                    log_probs, targets, output_lengths.cpu(), target_lengths
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["grad_clip"]
            )

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), n=features.size(0))

            if step % cfg["training"]["log_every_n_steps"] == 0:
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        scheduler.step()

        dev_loss, wer, cer = evaluate(
            model, dev_loader, device, criterion
        )

        print(
            f"Epoch {epoch}: train_loss={loss_meter.avg:.4f} "
            f"dev_loss={dev_loss:.4f} WER={wer:.3f} CER={cer:.3f}"
        )

        history["train_loss"].append(loss_meter.avg)
        history["dev_loss"].append(dev_loss)
        history["wer"].append(wer)
        history["cer"].append(cer)

        save_history(history, ckpt_dir / "history.json")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "cfg": cfg,
            },
            ckpt_dir / "last.pt",
        )

        if wer < best_wer:
            best_wer = wer

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "cfg": cfg,
                },
                ckpt_dir / "best.pt",
            )

            print(
                f"  -> new best WER {wer:.3f}, "
                "saved checkpoints/best.pt"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, eval_only=args.eval_only, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
