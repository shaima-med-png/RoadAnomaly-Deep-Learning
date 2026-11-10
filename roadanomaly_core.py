"""Road anomaly detector: model code and inference pipeline, without any user interface.

Same architectures as notebooks/RoadAnomaly_complete.ipynb, so the exported roadanomaly_model.pt loads directly.
Used by streamlit_app.py.
"""
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import models


# ---------------------------------------------------------------- architectures
def conv_bn_lrelu(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.LeakyReLU(0.1, inplace=True))


class BaselineCNN(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        layers, c_in = [], 3
        for c in widths:
            layers += [conv_bn_lrelu(c_in, c), conv_bn_lrelu(c, c), nn.MaxPool2d(2)]
            c_in = c
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Sequential(conv_bn_lrelu(c_in, 64), nn.Dropout2d(0.1), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        return F.interpolate(self.classifier(self.encoder(x)), size=x.shape[-2:], mode="bilinear", align_corners=False)

    def gradcam_layer(self):
        return self.encoder[-2]


class ResNet18Encoder(nn.Module):
    channels = [64, 64, 128, 256, 512]

    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=None)                    # trained weights come from the exported file
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool, self.layer1, self.layer2, self.layer3, self.layer4 = r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return [x0, x1, x2, x3, self.layer4(x3)]

    def gradcam_layer(self):
        return self.layer4


class EfficientNetB1Encoder(nn.Module):
    channels = [16, 24, 40, 112, 320]
    taps = (1, 2, 3, 5, 7)

    def __init__(self):
        super().__init__()
        self.features = models.efficientnet_b1(weights=None).features[:8]

    def forward(self, x):
        out = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.taps:
                out.append(x)
        return out

    def gradcam_layer(self):
        return self.features[7]


class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.conv = nn.Sequential(conv_bn_lrelu(c_in + c_skip, c_out), conv_bn_lrelu(c_out, c_out))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class TransferUNet(nn.Module):
    def __init__(self, encoder, dec=(256, 128, 64, 32)):
        super().__init__()
        self.encoder = encoder
        ch = encoder.channels
        self.ups = nn.ModuleList([UpBlock(ch[4], ch[3], dec[0]), UpBlock(dec[0], ch[2], dec[1]),
                                  UpBlock(dec[1], ch[1], dec[2]), UpBlock(dec[2], ch[0], dec[3])])
        self.head = nn.Sequential(conv_bn_lrelu(dec[3], 16), nn.Conv2d(16, 1, 1))

    def forward(self, x):
        feats = self.encoder(x)
        y = feats[-1]
        for up, skip in zip(self.ups, reversed(feats[:-1])):
            y = up(y, skip)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(y)

    def gradcam_layer(self):
        return self.encoder.gradcam_layer()


BUILDERS = {"baseline_cnn": BaselineCNN,
            "resnet18_unet": lambda: TransferUNet(ResNet18Encoder()),
            "efficientnet_b1_unet": lambda: TransferUNet(EfficientNetB1Encoder())}


# ---------------------------------------------------------------- detector
class Detector:
    """Loads the exported package and runs the same pipeline as section 7.3 of the notebook."""

    def __init__(self, model_file, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pkg = torch.load(model_file, map_location=self.device, weights_only=True)   # safe loading
        self.model = BUILDERS[self.pkg["arch"]]().to(self.device).eval()
        self.model.load_state_dict(self.pkg["state_dict"])
        self.h, self.w = self.pkg["input_size"]
        self.mean = torch.tensor(self.pkg["mean"]).view(3, 1, 1)
        self.std = torch.tensor(self.pkg["std"]).view(3, 1, 1)
        self.threshold = float(self.pkg["threshold"])
        self.name = self.pkg["model_name"]
        self.arch = self.pkg["arch"]
        self.metrics = self.pkg.get("test_metrics", {})
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.file_mb = Path(model_file).stat().st_size / 1e6

    def to_input(self, rgb):
        # same preprocessing as in training: resize to 512x288, scale to [0, 1], ImageNet normalisation
        x = torch.from_numpy(np.array(rgb.resize((self.w, self.h), Image.BILINEAR))).permute(2, 0, 1).float() / 255.0
        return ((x - self.mean) / self.std)[None].to(self.device)

    @torch.no_grad()
    def score_map(self, rgb):
        # anomaly probability of every pixel, at the original image size, and the inference time in ms
        t0 = time.time()
        prob = torch.sigmoid(self.model(self.to_input(rgb)))
        prob = F.interpolate(prob, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False)
        return prob[0, 0].cpu().numpy(), (time.time() - t0) * 1000

    def grad_cam(self, rgb, threshold):
        # which regions of the last encoder block drive the anomaly score of the detected region
        store = {}

        def hook(module, inputs, output):
            store["act"] = output
            output.register_hook(lambda g: store.__setitem__("grad", g))

        handle = self.model.gradcam_layer().register_forward_hook(hook)
        try:
            with torch.enable_grad():
                self.model.zero_grad()
                logits = self.model(self.to_input(rgb))
                region = (torch.sigmoid(logits) >= threshold).float()
                if region.sum() < 1:                          # nothing detected: explain the whole map
                    region = torch.ones_like(region)
                ((logits * region).sum() / region.sum()).backward()
        finally:
            handle.remove()
        weights = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False)[0, 0]
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


# ---------------------------------------------------------------- alert and drawing (numpy only)
_STOPS = np.array([[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]]) / 255.0


def colormap(v):
    # magma-like colours: dark = 0, light yellow = 1
    v = np.clip(v, 0, 1) * (len(_STOPS) - 1)
    i = np.minimum(v.astype(int), len(_STOPS) - 2)
    t = (v - i)[..., None]
    return (1 - t) * _STOPS[i] + t * _STOPS[i + 1]


def alert_from(prob, threshold):
    # image-level alert: objects = connected regions above the threshold, ignoring specks < 0.05 % of the image
    mask = prob >= threshold
    labels, _ = ndi.label(mask)
    objects = []
    for k, sl in enumerate(ndi.find_objects(labels), start=1):
        region = labels[sl] == k
        if region.sum() < 0.0005 * mask.size:
            continue
        ys, xs = np.nonzero(region)
        cy, cx = (ys.mean() + sl[0].start) / mask.shape[0], (xs.mean() + sl[1].start) / mask.shape[1]
        objects.append({"area (%)": round(float(100 * region.sum() / mask.size), 2),
                        "position": f"{['left', 'centre', 'right'][min(int(cx * 3), 2)]}, "
                                    f"{['far', 'middle', 'near'][min(int(cy * 3), 2)]}",
                        "box (x0, y0, x1, y1)": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)],
                        "max score": round(float(prob[sl][region].max()), 3),
                        "mean score": round(float(prob[sl][region].mean()), 3)})
    objects.sort(key=lambda o: -o["area (%)"])
    info = {"alert": len(objects) > 0, "objects": len(objects),
            "anomaly area (%)": round(100 * float(mask.mean()), 2),
            "main position": objects[0]["position"] if objects else "-",
            "max score": round(float(prob.max()), 3),
            "threshold": round(float(threshold), 2), "details": objects}
    return mask, info


def overlay(rgb, mask, objects=None):
    # anomaly pixels in red; with objects, a numbered box around each detected obstacle
    vis = np.array(rgb).astype(np.float32) / 255.0
    vis[mask] = 0.45 * vis[mask] + 0.55 * np.array([1.0, 0.1, 0.1])
    out = Image.fromarray((vis * 255).astype(np.uint8))
    if objects:
        draw = ImageDraw.Draw(out)
        lw = max(2, round(out.width / 400))
        size = max(14, round(out.width / 55))                 # label size follows the image size
        try:
            font = ImageFont.load_default(size=size)          # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()
        for k, o in enumerate(objects, start=1):
            x0, y0, x1, y1 = o["box (x0, y0, x1, y1)"]
            draw.rectangle([x0 - lw, y0 - lw, x1 + lw, y1 + lw], outline=(255, 214, 0), width=lw)
            tl, tt, tr, tb = draw.textbbox((0, 0), f"#{k}", font=font)
            tag_w, tag_h = tr - tl + 10, tb - tt + 8
            ty = y0 - lw - tag_h if y0 - lw - tag_h >= 0 else y1 + lw
            draw.rectangle([x0 - lw, ty, x0 - lw + tag_w, ty + tag_h], fill=(255, 214, 0))
            draw.text((x0 - lw + 5 - tl, ty + 4 - tt), f"#{k}", fill=(0, 0, 0), font=font)
    return out


def crop(rgb, box, margin=0.25, min_side=64):
    # zoom on one object, with some context around it
    x0, y0, x1, y1 = box
    mx, my = max(int((x1 - x0) * margin), min_side // 2), max(int((y1 - y0) * margin), min_side // 2)
    return rgb.crop((max(x0 - mx, 0), max(y0 - my, 0), min(x1 + mx, rgb.width), min(y1 + my, rgb.height)))


def heat_image(values, rgb=None):
    # colour map of a [0, 1] array, blended with the photo when one is given (for Grad-CAM)
    col = colormap(values)
    if rgb is not None:
        col = 0.5 * np.array(rgb).astype(np.float32) / 255.0 + 0.5 * col
    return Image.fromarray((col * 255).astype(np.uint8))
