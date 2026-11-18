"""Diagnose the shortcut: image size (a proxy for which hospital/source a scan came from) per class.

Usage: python analyze_sources.py [--model models/resnet50_brain_mri.pt]

Prints how many images per split/class are 512x512 vs other sizes, then shows how
a model does on the test gliomas from each group.
"""
import argparse
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from train import DATA, eval_tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/resnet50_brain_mri.pt")
    args = ap.parse_args()

    print("Share of images that are exactly 512x512 (one source):")
    for split in ["Training", "Testing"]:
        for c in sorted(p.name for p in (DATA / split).iterdir()):
            files = list((DATA / split / c).iterdir())
            n512 = sum(Image.open(f).size == (512, 512) for f in files)
            print(f"  {split:<9}{c:<11}{n512:>5}/{len(files):<5} ({n512 / len(files):.0%})")

    from gradcam import load
    model, classes, device = load(args.model)
    results = Counter()
    with torch.no_grad():
        for f in sorted((DATA / "Testing" / "glioma").iterdir()):
            im = Image.open(f).convert("RGB")
            group = "512x512" if im.size == (512, 512) else "other size"
            pred = classes[model(eval_tf(im).unsqueeze(0).to(device)).argmax(1).item()]
            results[(group, pred)] += 1

    print(f"\nTest gliomas, predictions of {args.model}:")
    for group in ["512x512", "other size"]:
        row = {p: results[(group, p)] for p in classes}
        total = sum(row.values())
        print(f"  {group:<11} n={total:<4} correct {row['glioma'] / total:.1%}   " +
              "  ".join(f"{p}:{n}" for p, n in row.items()))


if __name__ == "__main__":
    main()
