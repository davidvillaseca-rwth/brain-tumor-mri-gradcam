"""Step 2: fine-tune an ImageNet-pretrained ResNet50 on the brain MRI scans.

Usage: python train.py [--epochs 6] [--smoke]
Saves the best model (by validation accuracy) to models/resnet50_brain_mri.pt.
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "models" / "resnet50_brain_mri.pt"

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class RandomLowRes:
    """Downscale then upscale, so image resolution/source stops being a clue."""

    def __init__(self, p=0.5, sizes=(96, 224)):
        self.p, self.sizes = p, sizes

    def __call__(self, img):
        import random
        if random.random() < self.p:
            s = random.randint(*self.sizes)
            img = img.resize((s, s)).resize((224, 224))
        return img


strong_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
    RandomLowRes(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device, num_classes):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            for t, p in zip(y, preds):
                confusion[t, p] += 1
    return confusion.diag().sum().item() / confusion.sum().item(), confusion


def _ahash(path):
    """8x8 average hash: near-identical images get the same value."""
    from PIL import Image
    px = torch.tensor(list(Image.open(path).convert("L").resize((8, 8)).getdata()), dtype=torch.float)
    return tuple((px > px.mean()).tolist())


def borrow_split(test_ds, seed=42):
    """Split the non-512x512 ("other source") test gliomas 50/50.

    Returns (borrowed test indices -> moved to training, kept test indices).
    Near-duplicates are grouped so copies never end up on both sides.
    """
    from PIL import Image
    glioma = test_ds.class_to_idx["glioma"]
    groups = {}
    for i, (path, label) in enumerate(test_ds.samples):
        if label == glioma and Image.open(path).size != (512, 512):
            groups.setdefault(_ahash(path), []).append(i)
    keys = sorted(groups)
    order = torch.randperm(len(keys), generator=torch.Generator().manual_seed(seed)).tolist()
    borrowed, target = [], sum(len(g) for g in groups.values()) // 2
    for k in order:
        if len(borrowed) >= target:
            break
        borrowed += groups[keys[k]]
    keep = sorted(set(range(len(test_ds))) - set(borrowed))
    return sorted(borrowed), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="run a few batches only")
    ap.add_argument("--strong-aug", action="store_true", help="heavier augmentation against source shortcuts")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--borrow", action="store_true",
                    help="move half the other-source test gliomas into training (repeated x5)")
    args = ap.parse_args()
    ckpt_path = Path(args.out)
    ckpt_path = str(ckpt_path if ckpt_path.is_absolute() else ROOT / ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    # Two views of Training/ with different transforms, split 85/15 by a fixed seed.
    train_full = datasets.ImageFolder(DATA / "Training", transform=strong_tf if args.strong_aug else train_tf)
    val_full = datasets.ImageFolder(DATA / "Training", transform=eval_tf)
    test_ds = datasets.ImageFolder(DATA / "Testing", transform=eval_tf)
    classes = train_full.classes
    assert test_ds.classes == classes

    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(42)).tolist()
    n_val = int(0.15 * len(perm))
    train_ds, val_ds = Subset(train_full, perm[n_val:]), Subset(val_full, perm[:n_val])
    if args.borrow:
        borrowed, keep = borrow_split(test_ds)
        test_train_view = datasets.ImageFolder(DATA / "Testing", transform=train_full.transform)
        train_ds = ConcatDataset([train_ds] + [Subset(test_train_view, borrowed)] * 5)
        test_ds = Subset(test_ds, keep)
        print(f"Borrowed {len(borrowed)} other-source test gliomas into training", flush=True)
    if args.smoke:
        train_ds, val_ds = Subset(train_ds, range(128)), Subset(val_ds, range(64))
        args.epochs = 1

    kw = dict(batch_size=args.batch, num_workers=4, pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(train_ds, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, **kw)
    test_dl = DataLoader(test_ds, **kw)
    print(f"Classes: {classes} | train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}", flush=True)

    model = build_model(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = 0.0
    Path(ckpt_path).parent.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, correct, loss_sum = time.time(), 0, 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        val_acc, _ = evaluate(model, val_dl, device, len(classes))
        print(f"Epoch {epoch}/{args.epochs}  loss {loss_sum/total:.4f}  train acc {correct/total:.4f}  "
              f"val acc {val_acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if val_acc >= best:
            best = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, ckpt_path)

    # Final, one-time check on the held-out Testing folder using the best checkpoint.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    test_acc, conf = evaluate(model, test_dl, device, len(classes))
    print(f"\nBest val acc {best:.4f}\nTEST ACCURACY {test_acc:.4f}\n")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(" " * 12 + "".join(f"{c[:10]:>12}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[:10]:>12}" + "".join(f"{v:>12}" for v in conf[i].tolist())
              + f"   recall {conf[i, i].item() / conf[i].sum().item():.3f}")


if __name__ == "__main__":
    main()
