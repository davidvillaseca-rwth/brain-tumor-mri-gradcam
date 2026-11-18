"""Grad-CAM heatmaps: where in the scan does the model look?

Usage:
  python gradcam.py                          # grid of unseen test scans -> gradcam_grid.png
  python gradcam.py scan1.jpg scan2.png      # your own scans -> gradcam_custom.png
  python gradcam.py --model models/resnet50_brain_mri.pt --out docs/images/x.png
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets

from train import borrow_split, build_model, eval_tf

ROOT = Path(__file__).parent
DEFAULT_CKPT = ROOT / "models" / "resnet50_borrow.pt"  # strong aug + borrowed gliomas (best model)


class GradCAM:
    """Grad-CAM on the last bottleneck block of ResNet50's layer4 (7x7 feature map)."""

    def __init__(self, model):
        self.model = model
        self.acts = None
        model.layer4[-1].register_forward_hook(lambda m, i, o: setattr(self, "acts", o))

    def __call__(self, x, target=None):
        logits = self.model(x)
        probs = logits.softmax(1)[0]
        cls = int(probs.argmax()) if target is None else target
        grads = torch.autograd.grad(logits[0, cls], self.acts)[0]      # d(score)/d(features): (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)                 # one importance weight per channel
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))       # weighted sum, keep positive evidence
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cls, probs[cls].item(), cam.detach().cpu().numpy()


def load(ckpt_path=DEFAULT_CKPT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"], device


def explain(cam_fn, path, device):
    img = Image.open(path).convert("RGB")
    cls, conf, cam = cam_fn(eval_tf(img).unsqueeze(0).to(device))
    shown = np.asarray(img.resize((224, 224))) / 255.0
    return shown, cam, cls, conf


def plot(items, out, cols=4):
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (img, cam, title, ok) in zip(np.atleast_1d(axes).ravel(), items):
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.set_title(title, fontsize=9, color={True: "green", False: "red", None: "black"}[ok])
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("Saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scans", nargs="*", help="your own scan images (optional)")
    ap.add_argument("--model", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, classes, device = load(args.model)
    cam_fn = GradCAM(model)
    items = []

    if args.scans:  # user-supplied scans
        for p in args.scans:
            img, cam, cls, conf = explain(cam_fn, p, device)
            items.append((img, cam, f"{Path(p).name}\npred: {classes[cls]} ({conf:.0%})", None))
        plot(items, args.out or ROOT / "gradcam_custom.png")
        return

    # Skip the test gliomas that were borrowed into training (see train.py --borrow).
    test = datasets.ImageFolder(ROOT / "data" / "Testing")
    seen = {Path(test.samples[i][0]).resolve() for i in borrow_split(test)[0]}

    # 3 random correct scans per class, plus the first 4 mistakes found.
    rng = random.Random(0)
    mistakes = []
    for c in classes:
        files = [f for f in sorted((ROOT / "data" / "Testing" / c).iterdir()) if f.resolve() not in seen]
        rng.shuffle(files)
        shown = 0
        for f in files:
            img, cam, cls, conf = explain(cam_fn, f, device)
            ok = classes[cls] == c
            entry = (img, cam, f"true: {c}\npred: {classes[cls]} ({conf:.0%})", ok)
            if ok and shown < 3:
                items.append(entry)
                shown += 1
            elif not ok and len(mistakes) < 4:
                mistakes.append(entry)
            if shown >= 3 and len(mistakes) >= 4:
                break
    plot(items + mistakes, args.out or ROOT / "gradcam_grid.png")
    print(f"{len(mistakes)} misclassified examples included (red titles).")


if __name__ == "__main__":
    main()
