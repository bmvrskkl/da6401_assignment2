
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from PIL import Image

from multitask import MultiTaskPerceptionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

norm = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TRIMAP_PALETTE = {0: [0,0,255], 1: [0,255,0], 2: [255,0,0]}

def mask_to_rgb(mask_np):
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for cls, color in TRIMAP_PALETTE.items():
        rgb[mask_np == cls] = color
    return rgb


def run_inference(image_path: str):
    model = MultiTaskPerceptionModel().to(DEVICE)
    model.eval()

    img_pil  = Image.open(image_path).convert("RGB")
    img_t    = norm(img_pil).unsqueeze(0).to(DEVICE)
    img_disp = np.array(img_pil.resize((224, 224)))

    with torch.no_grad():
        cls_logits, bbox_pred, seg_logits = model(img_t)

    pred_class = cls_logits.argmax(1).item()
    pred_box   = bbox_pred[0].cpu().tolist()   # [cx, cy, w, h] pixel space
    pred_mask  = seg_logits.argmax(1)[0].cpu().numpy()

    print(f"Predicted class: {pred_class}")
    print(f"Bounding box:    {pred_box}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Classification + BBox
    axes[0].imshow(img_disp)
    cx, cy, w, h = pred_box
    rect = patches.Rectangle(
        (cx - w/2, cy - h/2), w, h,
        linewidth=2, edgecolor="red", facecolor="none"
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f"Class {pred_class}")
    axes[0].axis("off")

    # Original
    axes[1].imshow(img_disp)
    axes[1].set_title("Original")
    axes[1].axis("off")

    # Segmentation
    axes[2].imshow(mask_to_rgb(pred_mask))
    axes[2].set_title("Segmentation")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("inference_output.png", dpi=100)
    print("Saved inference_output.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()
    run_inference(args.image)
