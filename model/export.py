"""Export the best U-Net checkpoint to ONNX for CPU inference in the worker.

The worker runs on a cron schedule (GitHub Actions now, Lambda at step 5),
so the model must run without torch — onnxruntime CPU does a 512px frame in
~1-2 s. Verifies torch vs onnxruntime agreement before writing.

Usage:
    python -m model.export            # -> checkpoints/unet.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from model.train import CHECKPOINT_DIR, build_model

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=CHECKPOINT_DIR / "unet.onnx")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    checkpoint = torch.load(CHECKPOINT_DIR / "unet_best.pt",
                            map_location="cpu", weights_only=True)
    model = build_model(checkpoint["encoder"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    example = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model, example, args.out,
                      input_names=["image"], output_names=["logits"],
                      dynamo=False)

    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        want = model(example).numpy()
    got = session.run(None, {"image": example.numpy()})[0]
    max_diff = float(np.abs(want - got).max())
    agree = float((want.argmax(1) == got.argmax(1)).mean())
    log.info("torch vs onnx: max |logit diff| %.2e, argmax agreement %.4f",
             max_diff, agree)
    if agree < 0.999:
        raise SystemExit("ONNX export disagrees with torch — not shipping it")
    log.info("exported %s (%.1f MB, epoch %d)", args.out,
             args.out.stat().st_size / 1e6, checkpoint["epoch"])


if __name__ == "__main__":
    main()
