"""Train the U-Net segmenter (step 2, the model that must beat the baseline).

Architecture: smp.Unet with an ImageNet-pretrained encoder, 3 input channels
(AIA 171/193/304), 3 classes (bg / CH / AR). Loss is CE + Dice — CE alone
collapses under the ~95/4/1 class imbalance.

Selection: best mean(CH, AR) IoU on the val split (computed inside the disk,
same protocol as the baseline and the final test report).

Usage:
    python -m model.train --epochs 40
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

from model.data import Sample, load_split
from model.metrics import iou_per_class

log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")


class SolarDataset(Dataset):
    def __init__(self, samples: list[Sample], augment: bool):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = s.image.astype(np.float32) / 255.0
        mask = s.mask.astype(np.int64)
        if self.augment:
            if np.random.rand() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.5:
                image, mask = image[::-1], mask[::-1]
            k = np.random.randint(4)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        return (
            torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1),
            torch.from_numpy(np.ascontiguousarray(mask)),
        )


def build_model(encoder: str = "resnet18") -> torch.nn.Module:
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=3, classes=3)


@torch.no_grad()
def validate(model, samples: list[Sample], device) -> dict[str, float]:
    model.eval()
    preds = []
    for s in samples:
        x = torch.from_numpy(s.image.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None].to(device)
        preds.append(model(x).argmax(1)[0].cpu().numpy().astype(np.uint8))
    return iou_per_class(preds, [s.mask for s in samples],
                         [s.disk for s in samples])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder", default="resnet18")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info("device: %s", device)

    train_samples = load_split("train")
    val_samples = load_split("val")
    log.info("train %d / val %d samples", len(train_samples), len(val_samples))
    loader = DataLoader(SolarDataset(train_samples, augment=True),
                        batch_size=args.batch_size, shuffle=True)

    model = build_model(args.encoder).to(device)
    ce = torch.nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss(mode="multiclass")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_score, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = ce(logits, y) + dice(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val_iou = validate(model, val_samples, device)
        score = float(np.mean(list(val_iou.values())))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "val_iou": val_iou})
        marker = ""
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(),
                        "encoder": args.encoder, "epoch": epoch,
                        "val_iou": val_iou},
                       CHECKPOINT_DIR / "unet_best.pt")
            marker = "  <- best"
        log.info("epoch %02d  loss %.4f  val IoU CH %.3f AR %.3f%s",
                 epoch, np.mean(losses), val_iou["coronal_hole"],
                 val_iou["active_region"], marker)

    (CHECKPOINT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    log.info("best mean val IoU: %.3f — checkpoint %s",
             best_score, CHECKPOINT_DIR / "unet_best.pt")


if __name__ == "__main__":
    main()
