"""Score every saved model on the same test images (the ones never borrowed into training)."""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from train import DATA, borrow_split, build_model, eval_tf

MODELS = {
    "original": "models/resnet50_brain_mri.pt",
    "strong aug": "models/resnet50_strongaug.pt",
    "strong aug + borrow": "models/resnet50_borrow.pt",
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = datasets.ImageFolder(DATA / "Testing", transform=eval_tf)
    _, keep = borrow_split(test)
    glioma = test.class_to_idx["glioma"]
    other_src = {i for i in keep if test.samples[i][1] == glioma
                 and Image.open(test.samples[i][0]).size != (512, 512)}
    loader = DataLoader(Subset(test, keep), batch_size=64, num_workers=4)
    print(f"Shared test set: {len(keep)} images ({len(other_src)} other-source gliomas)\n")
    print(f"{'model':<22}{'accuracy':>10}{'glioma':>9}{'other-src':>11}{'gl->notumor':>13}")

    for name, path in MODELS.items():
        if not Path(path).exists():
            continue
        ckpt = torch.load(path, map_location=device)
        model = build_model(len(ckpt["classes"]))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                preds += model(x.to(device)).argmax(1).cpu().tolist()
        labels = [test.samples[i][1] for i in keep]
        notumor = test.class_to_idx["notumor"]
        acc = sum(p == y for p, y in zip(preds, labels)) / len(keep)
        gl = [(p, i) for p, i, y in zip(preds, keep, labels) if y == glioma]
        gl_rec = sum(p == glioma for p, _ in gl) / len(gl)
        os_rec = sum(p == glioma for p, i in gl if i in other_src) / len(other_src)
        to_no = sum(p == notumor for p, _ in gl)
        print(f"{name:<22}{acc:>10.1%}{gl_rec:>9.1%}{os_rec:>11.1%}{to_no:>13}")


if __name__ == "__main__":
    main()
