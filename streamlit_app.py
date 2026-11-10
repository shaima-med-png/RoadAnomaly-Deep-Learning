"""Road Anomaly Detector: Streamlit web application (deployment of notebooks/RoadAnomaly_complete.ipynb).

Pages
  Detect             UC1-UC4, UC6: alert, numbered obstacles, score map, Grad-CAM, per-object details, JSON export
  Batch inspection   UC5: many images at once, CSV report, gallery of alerts
  Model & choice     UC7: what the model is, why it was chosen, how it was tested, limits
  Admin              UC8: password-protected dashboard, settings, model reload, usage log

Secrets (Streamlit Cloud: App settings -> Secrets)
  MODEL_URL       direct link to roadanomaly_model.pt (for example a GitHub release asset)
  ADMIN_PASSWORD  password of the Admin page (the page stays locked when it is not set)

Run locally:   pip install -r requirements.txt   then   streamlit run streamlit_app.py
"""
import hashlib
import hmac
import html
import io
import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from roadanomaly_core import Detector, alert_from, crop, heat_image, overlay

HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / "roadanomaly_model.pt"
EXAMPLES = {p.stem: p for p in sorted([*(HERE / "examples").glob("*.jpg"), *HERE.glob("*.jpg")],
                                      key=lambda p: (p.stem.startswith("new_"), p.stem))}
EXAMPLE_INFO = {"animals11": "Pheasant (RoadAnomaly test)", "animals15": "Horse and tram (RoadAnomaly test)",
                "cones02": "Cones, open manhole (RoadAnomaly test)", "vehicle01": "Bicycle with hay (RoadAnomaly test)",
                "new_giraffe": "Giraffe (new photo, never seen)", "new_sloth": "Sloth (new photo, never seen)",
                "new_camels": "Camels (new photo, never seen)", "new_elephant": "Elephant (new photo, never seen)"}
DATA_DIR = HERE / "app_data"
LOG_FILE = DATA_DIR / "usage_log.csv"
SETTINGS_FILE = DATA_DIR / "settings.json"
LOG_COLUMNS = ["time (UTC)", "page", "image", "alert", "objects", "anomaly area (%)", "max score", "threshold",
               "latency (ms)"]
DEFAULT_SETTINGS = {"default_threshold": None, "gradcam_enabled": True, "max_batch": 30, "announcement": ""}

st.set_page_config(page_title="Road Anomaly Detector", page_icon="🚧", layout="wide")

st.markdown("""
<style>
.ra-hero {padding: 4px 0 10px;}
.ra-hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: -0.02em;}
.ra-hero p {margin: 4px 0 0; opacity: 0.8; font-size: 1.02rem;}
.ra-pill {display: inline-block; padding: 2px 10px; margin: 6px 6px 0 0; border-radius: 999px; font-size: 0.8rem;
          background: rgba(127, 127, 127, 0.15);}
.ra-banner {padding: 14px 18px; border-radius: 10px; margin: 4px 0 14px; font-size: 1.05rem; line-height: 1.5;}
.ra-banner b {font-size: 1.15rem;}
.ra-alert {background: rgba(220, 38, 38, 0.12); border-left: 6px solid #dc2626;}
.ra-ok {background: rgba(22, 163, 74, 0.12); border-left: 6px solid #16a34a;}
.ra-note {background: rgba(37, 99, 235, 0.10); border-left: 6px solid #2563eb;}
.ra-warn {background: rgba(217, 119, 6, 0.12); border-left: 6px solid #d97706;}
.ra-card {padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(127, 127, 127, 0.25); margin-bottom: 10px;}
.ra-card h4 {margin: 0 0 4px;}
.ra-small {font-size: 0.85rem; opacity: 0.8;}
.ra-kicker {text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.75rem; opacity: 0.7;}
.ra-feature {padding: 16px 18px; border-radius: 12px; border: 1px solid rgba(127, 127, 127, 0.25); min-height: 128px;
             margin-bottom: 8px;}
.ra-feature h4 {margin: 2px 0 6px;}
.ra-feature p {margin: 0; opacity: 0.85; font-size: 0.95rem;}
.ra-footer {margin-top: 40px; padding-top: 12px; border-top: 1px solid rgba(127, 127, 127, 0.25); font-size: 0.8rem;
            opacity: 0.7;}
</style>
""", unsafe_allow_html=True)


def banner(text, kind="note"):
    st.markdown(f'<div class="ra-banner ra-{kind}">{text}</div>', unsafe_allow_html=True)


def secret(name):
    try:
        value = st.secrets.get(name, "")
    except Exception:                                         # no secrets file at all
        value = ""
    return value or os.environ.get(name, "")


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading the model…")
def load_detector():
    t0 = time.time()
    if MODEL_FILE.exists():                                   # a model file next to the app has priority
        path, source = MODEL_FILE, "local file roadanomaly_model.pt"
    else:
        url = secret("MODEL_URL")
        if not url:
            return None, "MODEL_URL secret is not set and roadanomaly_model.pt is not next to the app", 0.0
        # one file per link: changing MODEL_URL (for example to a v2 release) downloads the new model
        path = HERE / f"model_{hashlib.md5(url.encode()).hexdigest()[:10]}.pt"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        source = url
    return Detector(path), source, time.time() - t0


det, MODEL_SOURCE, LOAD_SECONDS = load_detector()
if det is None:
    st.error(f"The model could not be loaded: {MODEL_SOURCE}. See README.md (MODEL_URL secret).")
    st.stop()


@st.cache_data(max_entries=24, show_spinner=False)
def scores_of(image_bytes):
    return det.score_map(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


@st.cache_data(max_entries=12, show_spinner=False)
def cam_of(image_bytes, threshold):
    return det.grad_cam(Image.open(io.BytesIO(image_bytes)).convert("RGB"), threshold)


@st.cache_data(show_spinner=False)
def thumb_of(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((360, 200))
    return img


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- shared state: settings and usage log
@st.cache_resource
def shared_store():
    store = {"lock": threading.Lock(), "settings": dict(DEFAULT_SETTINGS), "log": []}
    DATA_DIR.mkdir(exist_ok=True)
    try:
        store["settings"].update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    try:
        store["log"] = pd.read_csv(LOG_FILE).to_dict("records")[-5000:]
    except Exception:
        pass
    return store


STORE = shared_store()
SETTINGS = STORE["settings"]
DEFAULT_THR = float(SETTINGS["default_threshold"] if SETTINGS["default_threshold"] is not None else det.threshold)


def save_settings():
    with STORE["lock"]:
        SETTINGS_FILE.write_text(json.dumps(SETTINGS, indent=1), encoding="utf-8")


def log_event(page, name, image_bytes, info, latency, threshold):
    # one line per analysed image (not one per page refresh): images are never stored, only their file name
    key = (page, hashlib.md5(image_bytes).hexdigest(), round(threshold, 2))
    seen = st.session_state.setdefault("logged", set())
    if key in seen:
        return
    seen.add(key)
    row = {"time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "page": page, "image": name,
           "alert": bool(info["alert"]), "objects": info["objects"], "anomaly area (%)": info["anomaly area (%)"],
           "max score": info["max score"], "threshold": round(threshold, 2), "latency (ms)": round(latency, 1)}
    with STORE["lock"]:
        STORE["log"].append(row)
        new = not LOG_FILE.exists()
        pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(LOG_FILE, mode="a", header=new, index=False)


# ---------------------------------------------------------------- sidebar: navigation and sensitivity
PAGES = ["Home", "Detect", "Batch inspection", "Model & choice", "Admin"]
ICONS = {"Home": "🏠", "Detect": "🔎", "Batch inspection": "🗂️", "Model & choice": "🧠", "Admin": "🔒"}


def go_to(target):
    st.session_state["page"] = target                        # callback: runs before the next rerun


def pick(name):
    st.session_state["picked"] = name


with st.sidebar:
    st.markdown("### 🚧 Road Anomaly Detector")
    page = st.radio("Page", PAGES, key="page", label_visibility="collapsed", format_func=lambda p: f"{ICONS[p]}  {p}")
    threshold = DEFAULT_THR
    explain = False
    if page in ("Detect", "Batch inspection"):
        st.divider()
        st.markdown("**Sensitivity**")
        presets = {"High sensitivity: miss fewer obstacles": 0.25,
                   f"Recommended ({DEFAULT_THR:.2f})": round(DEFAULT_THR, 2),
                   "Few false alarms: confident detections only": 0.70}
        preset = st.radio("Preset", list(presets), index=1, label_visibility="collapsed")
        threshold = st.slider("Decision threshold", 0.05, 0.95, presets[preset], 0.01, key=f"thr_{preset}",
                              help="A pixel is an anomaly when its score is above this value. Lower = more "
                                   "sensitive: fewer missed obstacles, more false alarms.")
        if page == "Detect" and SETTINGS["gradcam_enabled"]:
            explain = st.toggle("Explain with Grad-CAM", value=True, help="Shows which regions the model relied on. "
                                "Takes about twice as long.")
    st.divider()
    st.caption(f"Model: **{det.name}**  \n{det.n_params / 1e6:.1f} M parameters · {det.file_mb:.1f} MB · "
               f"runs on {det.device.type.upper()}")

if SETTINGS["announcement"].strip():
    banner("📢 " + html.escape(SETTINGS["announcement"].strip()), "warn")


def hero(title, subtitle):
    st.markdown(f'<div class="ra-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def fmt(value, spec=".2f"):
    return "–" if value is None else format(value, spec)


# ================================================================ page: Home
def page_home():
    st.markdown('<div class="ra-kicker">Deep Learning project · pixel-wise anomaly segmentation</div>',
                unsafe_allow_html=True)
    hero("Road Anomaly Detector",
         "Finds unexpected obstacles on the road (animals, rocks, cones, lost tires, unusual vehicles) in a single "
         "photo, pixel by pixel, and raises an alert with their number, size and position.")
    m = det.metrics
    c = st.columns(5)
    c[0].metric("Test mean F1", fmt(m.get("mean_f1")))
    c[1].metric("Recall", fmt(m.get("recall")), help="Share of anomaly pixels that are found (safety).")
    c[2].metric("Precision", fmt(m.get("precision")), help="Share of alerted pixels that are real anomalies (trust).")
    c[3].metric("Training images", "40", help="Plus 10 for validation and 10 for the test: 60 RoadAnomaly images.")
    c[4].metric("Model size", f"{det.n_params / 1e6:.1f} M", help="Parameters of the deployed network.")
    st.write("")

    cards = [("🔎 Detect", "Analyse one photo: alert, numbered obstacles, score map, Grad-CAM explanation and a "
                          "comparison of three sensitivity levels.", "Detect"),
             ("🗂️ Batch inspection", "Check a whole inspection run at once and download a CSV report of the alerts.",
              "Batch inspection"),
             ("🧠 Model & choice", "How the model works, why it was chosen among six candidates and how it behaves "
                                  "on images it has never seen.", "Model & choice")]
    for col, (title, text, target) in zip(st.columns(3), cards):
        with col:
            st.markdown(f'<div class="ra-feature"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
            st.button(f"Open {target}", key=f"go_{target}", on_click=go_to, args=(target,),
                      type="primary" if target == "Detect" else "secondary")

    st.subheader("How it works")
    encoder = {"efficientnet_b1_unet": "EfficientNet-B1 encoder", "resnet18_unet": "ResNet18 encoder"}.get(det.arch, "CNN")
    st.graphviz_chart(f"""
digraph {{
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#e8f0fa", color="#1e58a0", fontname="Helvetica", fontsize=11];
  edge [color="#8a94a6"];
  photo [label="Road photo\\nany size", fillcolor="#ffffff"];
  prep [label="Resize to 512×288\\nImageNet normalisation"];
  enc [label="{encoder}\\npretrained on ImageNet"];
  dec [label="U-Net decoder\\nskip connections"];
  prob [label="Probability\\nfor every pixel"];
  thr [label="Threshold\\n(recommended {DEFAULT_THR:.2f})"];
  out [label="Numbered obstacles\\n+ alert", fillcolor="#fff2dc", color="#a85f00"];
  photo -> prep -> enc -> dec -> prob -> thr -> out;
}}""")

    left, right = st.columns(2)
    with left:
        st.markdown("**What it does well**\n\n"
                    f"- Mean F1 {fmt(m.get('mean_f1'))} and recall {fmt(m.get('recall'))} on the test images, "
                    "with only 40 training images (transfer learning from ImageNet).\n"
                    "- Keeps working on new photos it has never seen (external test on the SegmentMeIfYouCan benchmark).\n"
                    "- Explains its decisions (Grad-CAM) and lets the user choose the sensitivity.")
    with right:
        st.markdown("**Where to be careful**\n\n"
                    "- Very small or distant obstacles can be missed.\n"
                    "- People and ordinary vehicles are not labelled as anomalies but can trigger alerts.\n"
                    "- Built for daytime photos: night, heavy rain and video still need validation. It assists a "
                    "human operator; use in a real vehicle would require safety certification.")


# ================================================================ page: Detect
def page_detect():
    hero("Detect anomalies on the road",
         "Choose a demo image or upload your own road photo. Every pixel that looks like an unexpected obstacle is "
         "marked in red, each obstacle is numbered, and an alert tells you how many there are and where.")
    source = st.radio("Image", ["Demo images", "Upload a photo"], horizontal=True, key="source",
                      index=0 if EXAMPLES else 1, label_visibility="collapsed")
    image_bytes, name = None, None
    if source == "Upload a photo" or not EXAMPLES:
        up = st.file_uploader("Road image (JPG, PNG or WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            image_bytes, name = up.getvalue(), up.name
    else:
        names = list(EXAMPLES)
        picked = st.session_state.get("picked")
        picked = picked if picked in EXAMPLES else names[0]
        cols = st.columns(4)
        for k, n in enumerate(names):
            with cols[k % 4]:
                st.image(thumb_of(str(EXAMPLES[n])), caption=EXAMPLE_INFO.get(n, n))
                st.button("✓ Analysed below" if n == picked else "Analyse", key=f"pick_{n}", on_click=pick, args=(n,),
                          disabled=n == picked)
        image_bytes, name = EXAMPLES[picked].read_bytes(), EXAMPLES[picked].name
        st.caption("Test photos never used for training. Credits and licences: CREDITS.txt in the repository.")

    if image_bytes is None:
        banner("Upload a road image, or choose a test image, to start.", "note")
        return

    try:
        rgb = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        st.error("This file is not a readable image. Try a JPG, PNG or WEBP photo.")
        return
    with st.spinner("Detecting anomalies…"):
        prob, latency = scores_of(image_bytes)
    mask, info = alert_from(prob, threshold)
    log_event("detect", name, image_bytes, info, latency, threshold)
    objects = info["details"]

    if info["alert"]:
        n = info["objects"]
        banner(f"<b>⚠ ALERT: {n} unexpected obstacle{'s' if n > 1 else ''} on the road</b><br>"
               f"Largest one: {objects[0]['position']} · {objects[0]['area (%)']} % of the image · "
               f"confidence {objects[0]['max score']:.0%}", "alert")
    else:
        banner(f"<b>✓ No alert</b><br>No region above the threshold {threshold:.2f}. "
               f"Highest pixel score in the image: {info['max score']:.0%}.", "ok")

    c = st.columns(5)
    c[0].metric("Obstacles", info["objects"])
    c[1].metric("Anomaly area", f"{info['anomaly area (%)']} %")
    c[2].metric("Highest score", f"{info['max score']:.0%}")
    c[3].metric("Threshold", f"{threshold:.2f}")
    c[4].metric("Inference", f"{latency:.0f} ms")

    annotated = overlay(rgb, mask, objects)
    tabs = st.tabs(["Detection", "Score map", "Grad-CAM", "Sensitivity comparison", "Original"])
    with tabs[0]:
        st.image(annotated, caption="Anomaly pixels in red, obstacles in numbered yellow boxes")
        st.download_button("Download the annotated image (PNG)", png_bytes(annotated),
                           file_name=f"{Path(name).stem}_detection.png", mime="image/png")
    with tabs[1]:
        st.image(heat_image(prob), caption="Score of every pixel before the threshold: dark = normal, yellow = anomaly")
    with tabs[2]:
        if not SETTINGS["gradcam_enabled"]:
            st.info("Grad-CAM is disabled by the administrator.")
        elif not explain:
            st.info("Turn on **Explain with Grad-CAM** in the sidebar.")
        else:
            with st.spinner("Computing Grad-CAM…"):
                cam = cam_of(image_bytes, round(threshold, 2))
            st.image(heat_image(cam, rgb), caption="Grad-CAM: warm colours = regions the model relied on for this alert")
    with tabs[3]:
        st.caption("The same score map cut at three thresholds: lower = more sensitive (fewer misses, more false alarms).")
        cols = st.columns(3)
        for col, (label, t) in zip(cols, [("High sensitivity", 0.25), ("Recommended", DEFAULT_THR),
                                          ("Few false alarms", 0.70)]):
            m_t, i_t = alert_from(prob, t)
            col.image(overlay(rgb, m_t, i_t["details"]),
                      caption=f"{label} · threshold {t:.2f} · {i_t['objects']} obstacle(s) · {i_t['anomaly area (%)']} %")
    with tabs[4]:
        st.image(rgb, caption=f"{name} · {rgb.width}×{rgb.height}")

    with st.expander("How to read the results"):
        st.markdown("- **Red pixels**: score above the threshold.\n"
                    "- **Yellow boxes**: one per obstacle, numbered by size (#1 = largest).\n"
                    "- **Score map**: the probability of every pixel before the threshold "
                    "(dark = surely normal, light yellow = surely an anomaly).\n"
                    "- **Grad-CAM**: the image regions that drove the decision.\n"
                    "- Regions smaller than 0.05 % of the image are ignored (noise).")

    if objects:
        st.subheader("Detected obstacles")
        table = pd.DataFrame([{"#": k, "position": o["position"], "area (%)": o["area (%)"],
                               "max score": o["max score"], "mean score": o["mean score"],
                               "box (x0, y0, x1, y1)": str(o["box (x0, y0, x1, y1)"])}
                              for k, o in enumerate(objects, start=1)])
        st.dataframe(table, hide_index=True)
        cols = st.columns(4)
        for k, o in enumerate(objects[:8], start=1):
            cols[(k - 1) % 4].image(crop(rgb, o["box (x0, y0, x1, y1)"]),
                                    caption=f"#{k} · {o['position']} · {o['area (%)']} % · score {o['max score']:.0%}")
        if len(objects) > 8:
            st.caption(f"{len(objects) - 8} more obstacles are listed in the table.")

    report = {"model": det.name, "image": name, "image size": [rgb.width, rgb.height],
              "latency (ms)": round(latency, 1), **info}
    with st.expander("Export the alert for another system (JSON)"):
        st.json(report)
        st.download_button("Download the alert (JSON)", json.dumps(report, indent=2),
                           file_name=f"{Path(name).stem}_alert.json", mime="application/json")


# ================================================================ page: Batch inspection
def page_batch():
    hero("Batch inspection", "For a road inspection run: add many images at once and get one line per image, "
                             "the images with an alert and a CSV report.")
    max_batch = int(SETTINGS["max_batch"])
    ups = st.file_uploader(f"Road images (up to {max_batch})", type=["jpg", "jpeg", "png", "webp"],
                           accept_multiple_files=True)
    if ups and len(ups) > max_batch:
        st.warning(f"{len(ups)} images selected: only the first {max_batch} will be inspected (limit set by the administrator).")
        ups = ups[:max_batch]
    if not st.button("Inspect all images", type="primary", disabled=not ups):
        if not ups:
            banner("Add one or more road images, then press <b>Inspect all images</b>.", "note")
        return
    rows, flagged = [], []
    bar = st.progress(0.0, text="Inspecting…")
    for k, up in enumerate(ups, start=1):
        data = up.getvalue()
        try:
            rgb = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            rows.append({"image": up.name, "status": "unreadable file"})
            continue
        prob, latency = scores_of(data)
        mask, info = alert_from(prob, threshold)
        log_event("batch", up.name, data, info, latency, threshold)
        rows.append({"image": up.name, "status": "ALERT" if info["alert"] else "ok", "objects": info["objects"],
                     "anomaly area (%)": info["anomaly area (%)"], "main position": info["main position"],
                     "highest score": info["max score"], "latency (ms)": round(latency, 1)})
        if info["alert"]:
            flagged.append((overlay(rgb, mask, info["details"]),
                            f"{up.name}: {info['objects']} obstacle(s), {info['main position']}"))
        bar.progress(k / len(ups), text=f"Inspecting… {k}/{len(ups)}")
    bar.empty()
    table = pd.DataFrame(rows)
    n_alert = int((table["status"] == "ALERT").sum())
    c = st.columns(4)
    c[0].metric("Images inspected", len(table))
    c[1].metric("With an alert", n_alert)
    c[2].metric("Alert rate", f"{n_alert / max(len(table), 1):.0%}")
    if "latency (ms)" in table:
        c[3].metric("Mean inference", f"{table['latency (ms)'].mean():.0f} ms")
    st.dataframe(table, hide_index=True)
    st.download_button("Download the report (CSV)", table.to_csv(index=False), file_name="road_inspection_report.csv",
                       mime="text/csv", type="primary")
    if flagged:
        st.subheader("Images with an alert")
        cols = st.columns(3)
        for k, (img, cap) in enumerate(flagged):
            cols[k % 3].image(img, caption=cap)


# ================================================================ page: Model & choice
# notebook 1, run of 7 November 2026: the six models trained and compared on the same 10 test images
COMPARISON_7NOV = pd.DataFrame([
    ["Baseline CNN", "from scratch", 1.32, 3.85, 0.751, 0.563],
    ["Convolutional autoencoder", "from scratch, unsupervised", 0.87, 2.95, 0.592, 0.270],
    ["ResNet18-UNet (SGD)", "ImageNet-pretrained", 14.33, 7.81, 0.795, 0.630],
    ["ResNet18-UNet (Adam)", "ImageNet-pretrained", 14.33, 7.59, 0.842, 0.714],
    ["EfficientNet-B1-UNet (Adam)", "ImageNet-pretrained", 8.34, 13.50, 0.852, 0.735],
    ["ResNet18-UNet (RMSprop)", "ImageNet-pretrained", 14.33, 7.75, 0.879, 0.779],
], columns=["model", "family", "parameters (M)", "GPU ms / image", "test mean F1", "test Dice"])

# companion notebook RoadAnomaly_extra_data.ipynb: same architecture and recipe, tested on images never seen
EXTERNAL = pd.DataFrame([
    ["RoadAnomaly test (10)", "same kind of images as training", 0.864, 0.752, 0.806, 0.705],
    ["AnomalyTrack (10)", "new web photos: animals, objects, vehicles", 0.802, 0.651, 0.891, 0.513],
    ["ObstacleTrack (10)", "small obstacles filmed from a car", 0.912, 0.826, 0.904, 0.760],
], columns=["test set", "content", "mean F1", "Dice", "precision", "recall"])


def page_model():
    hero("The model and why it was chosen",
         "What runs behind the app, how it was selected among six candidates, and how far it can be trusted.")
    m = det.metrics
    st.markdown(f'<span class="ra-pill">{det.name}</span><span class="ra-pill">{det.arch}</span>'
                f'<span class="ra-pill">{det.n_params / 1e6:.1f} M parameters</span>'
                f'<span class="ra-pill">threshold {det.threshold:.3f}</span>'
                f'<span class="ra-pill">input {det.w}×{det.h}</span>', unsafe_allow_html=True)
    if m:
        st.write("")
        c = st.columns(5)
        for col, (label, key) in zip(c, [("mean F1", "mean_f1"), ("Dice (F1)", "dice"), ("precision", "precision"),
                                         ("recall", "recall"), ("IoU", "iou")]):
            if key in m:
                col.metric(label, f"{m[key]:.3f}")
        st.caption("Test metrics stored in the model file (10 RoadAnomaly test images, threshold chosen on the "
                   "validation set).")

    st.subheader("1 · What the model does")
    st.markdown(
        "The task is **binary semantic segmentation**: for every pixel, the probability that it belongs to an "
        "unexpected obstacle on the road (animal, rock, cone, lost tire, unusual vehicle). The pipeline is:\n\n"
        "1. resize the photo to 512×288 and normalise it like ImageNet;\n"
        "2. **encoder**: an EfficientNet-B1 network pretrained on 1.2 million ImageNet photos extracts features at 5 "
        "scales, from edges to whole objects;\n"
        "3. **decoder (U-Net)**: goes back to full resolution, step by step, reusing the encoder features of the "
        "same size (*skip connections*) to draw precise outlines;\n"
        "4. a sigmoid gives one probability per pixel; pixels above the threshold form the mask, and connected "
        "regions become numbered obstacles.")

    st.subheader("2 · Why this model")
    st.markdown("Six models were trained on the same 40 images and compared on the same 10 test images "
                "(notebook `RoadAnomaly_complete.ipynb`):")
    st.dataframe(COMPARISON_7NOV, hide_index=True)
    st.bar_chart(COMPARISON_7NOV.set_index("model")[["test mean F1", "test Dice"]], horizontal=True)
    st.caption("Run of 7 November 2026. A model that answers \"normal\" everywhere scores mean F1 0.475 and Dice 0.")
    st.markdown(
        "**What the comparison shows**\n\n"
        "- **Transfer learning is decisive.** With only 40 training images, the pretrained U-Nets beat the CNN "
        "trained from scratch by about +0.10 mean F1 and +0.15 Dice.\n"
        "- **The unsupervised autoencoder is not enough.** Its reconstruction error measures \"hard to redraw\", "
        "not \"obstacle\" (precision 0.24), so labelled masks are worth using.\n"
        "- **The two best encoders are statistically tied.** ResNet18-UNet won the run of 7 November (0.879) and "
        "EfficientNet-B1-UNet won the final run of 8 November "
        f"({m.get('mean_f1', 0.889):.3f} mean F1, {m.get('dice', 0.799):.3f} Dice, recall {m.get('recall', 0.783):.3f}). "
        "GPU training is not fully deterministic, and with 10 test images their confidence intervals overlap.\n\n"
        "**Why EfficientNet-B1-UNet was deployed**\n\n"
        "1. best test mean F1 of the final run, chosen by a rule set in advance (best mean F1 on the test set);\n"
        "2. its recall (0.78) meets the safety target of the project (≥ 0.75), which the earlier best model missed;\n"
        "3. **lighter**: 8.3 M parameters (33.8 MB) instead of 14.3 M for ResNet18-UNet, which matters on the free CPU "
        "server of this app;\n"
        "4. same input, same pipeline and same threshold logic as all the other candidates, so switching costs nothing.")
    if det.arch != "efficientnet_b1_unet":
        banner(f"Note: the model currently deployed is <b>{html.escape(det.name)}</b>; the explanation above "
               "describes the selection of 8 November.", "warn")

    st.subheader("3 · How it behaves on images it has never seen")
    st.markdown("A companion experiment (`RoadAnomaly_extra_data.ipynb`) retrained the same architecture with the "
                "same recipe and tested it on 20 real images from the **SegmentMeIfYouCan** benchmark, never used "
                "for training:")
    st.dataframe(EXTERNAL, hide_index=True)
    st.markdown(
        "- On **new web photos** (AnomalyTrack) the model stays reliable when it raises an alert (precision 0.89) "
        "but finds only about half of each object's pixels (recall 0.51): it sees the obstacle, not always all of it.\n"
        "- On **small obstacles filmed from a car** (ObstacleTrack) it finds most of them; the misses are objects "
        "of a few pixels once the image is reduced to 512×288.\n"
        "- On **snowy roads** (no masks, visual check) it raised almost no false alarm.\n"
        "- Adding 20 small-obstacle images to training raised recall on small obstacles (0.76 → 0.95) but lowered "
        "precision (0.90 → 0.71). With no clear overall gain, the deployed model was kept.")

    st.subheader("4 · Limits and responsible use")
    recall = m.get("recall")
    banner("This is a <b>decision-support prototype</b>, not a safety-certified component.<br>"
           "• Trained on 40 daytime web images (60 in the dataset, with validation and test): not validated at night, in heavy rain, on other cameras or on video.<br>"
           "• People and ordinary vehicles are not labelled as anomalies but can trigger alerts.<br>"
           + (f"• Test recall {recall:.2f}: about {100 * (1 - recall):.0f} % of the anomaly pixels are missed.<br>" if recall else "")
           + "• Very small or distant obstacles are the main weakness.", "warn")


# ================================================================ page: Admin
def page_admin():
    hero("Administration", "Usage dashboard, settings, model and request log. Restricted to the administrator.")
    expected = secret("ADMIN_PASSWORD")
    if not expected:
        banner("<b>The Admin page is locked.</b><br>Set an <code>ADMIN_PASSWORD</code> secret to enable it "
               "(Streamlit Cloud: <i>App settings → Secrets</i>), then reboot the app.", "warn")
        return

    if not st.session_state.get("is_admin"):
        locked_for = st.session_state.get("locked_until", 0) - time.time()
        if locked_for > 0:
            st.error(f"Too many wrong passwords. Try again in {locked_for:.0f} s.")
            return
        with st.form("login"):
            password = st.text_input("Administrator password", type="password")
            ok = st.form_submit_button("Sign in", type="primary")
        if ok:
            if hmac.compare_digest(password.encode(), expected.encode()):
                st.session_state["is_admin"] = True
                st.session_state["fails"] = 0
                st.rerun()
            else:
                st.session_state["fails"] = st.session_state.get("fails", 0) + 1
                if st.session_state["fails"] >= 5:
                    st.session_state["locked_until"] = time.time() + 60
                    st.session_state["fails"] = 0
                st.error("Wrong password.")
        return

    top = st.columns([4, 1])
    top[0].success("Signed in as administrator.")
    if top[1].button("Sign out"):
        st.session_state["is_admin"] = False
        st.rerun()

    log = pd.DataFrame(STORE["log"], columns=LOG_COLUMNS)
    t_dash, t_settings, t_model, t_log = st.tabs(["Dashboard", "Settings", "Model", "Request log"])

    with t_dash:
        if log.empty:
            st.info("No image analysed yet since the app started.")
        else:
            log["time (UTC)"] = pd.to_datetime(log["time (UTC)"])
            last_day = log[log["time (UTC)"] >= log["time (UTC)"].max() - pd.Timedelta(days=1)]
            alerts = log["alert"].astype(str).str.lower().eq("true")
            c = st.columns(5)
            c[0].metric("Images analysed", len(log))
            c[1].metric("Last 24 h", len(last_day))
            c[2].metric("Alert rate", f"{alerts.mean():.0%}")
            c[3].metric("Mean inference", f"{log['latency (ms)'].mean():.0f} ms")
            c[4].metric("Mean threshold used", f"{log['threshold'].mean():.2f}")
            per_day = log.assign(day=log["time (UTC)"].dt.date, alert=alerts).groupby("day")["alert"] \
                         .agg(analysed="count", alerts="sum")
            st.markdown("**Images analysed and alerts per day**")
            st.bar_chart(per_day)
            st.markdown("**Inference time (ms), most recent requests**")
            st.line_chart(log.tail(200).set_index("time (UTC)")["latency (ms)"])
            st.caption("Data kept on the app server since the last restart (Streamlit Community Cloud resets it on reboot).")

    with t_settings:
        with st.form("settings"):
            use_validated = st.checkbox(f"Use the validated threshold of the model ({det.threshold:.2f})",
                                        value=SETTINGS["default_threshold"] is None)
            thr = st.slider("Default threshold for users", 0.05, 0.95,
                            float(SETTINGS["default_threshold"] or round(det.threshold, 2)), 0.01,
                            help="Used by the 'Recommended' preset. Ignored when the validated threshold is used.")
            gradcam = st.toggle("Allow Grad-CAM explanations", value=bool(SETTINGS["gradcam_enabled"]),
                                help="Grad-CAM doubles the computation time on the free CPU server.")
            max_batch = st.number_input("Maximum images per batch", 1, 200, int(SETTINGS["max_batch"]))
            note = st.text_area("Message shown to all users (empty = none)", SETTINGS["announcement"], max_chars=300)
            if st.form_submit_button("Save settings", type="primary"):
                SETTINGS.update(default_threshold=None if use_validated else round(float(thr), 2),
                                gradcam_enabled=bool(gradcam), max_batch=int(max_batch), announcement=note)
                save_settings()
                st.toast("Settings saved.")
                st.rerun()

    with t_model:
        st.dataframe(pd.DataFrame({"value": {
            "model": det.name, "architecture": det.arch, "parameters": f"{det.n_params / 1e6:.2f} M",
            "file size": f"{det.file_mb:.1f} MB", "validated threshold": f"{det.threshold:.3f}",
            "input size": f"{det.w} × {det.h}", "device": det.device.type,
            "loaded from": MODEL_SOURCE, "load time": f"{LOAD_SECONDS:.1f} s"}}))
        if det.metrics:
            st.markdown("**Test metrics stored in the file**")
            st.dataframe(pd.DataFrame({"value": {k: round(v, 3) for k, v in det.metrics.items()}}))
        st.markdown("To deploy a new model: publish it in a GitHub release, change the `MODEL_URL` secret, then "
                    "press **Reload the model**.")
        if st.button("Reload the model"):
            load_detector.clear()
            scores_of.clear()
            cam_of.clear()
            st.rerun()

    with t_log:
        if log.empty:
            st.info("The log is empty.")
        else:
            st.dataframe(log.sort_values("time (UTC)", ascending=False).head(300), hide_index=True)
            st.download_button("Download the full log (CSV)", log.to_csv(index=False), file_name="usage_log.csv",
                               mime="text/csv")
        st.caption("The log stores the file name and the result of each analysis, never the image itself.")
        confirm = st.checkbox("I want to delete the whole log")
        if st.button("Delete the log", disabled=not confirm):
            with STORE["lock"]:
                STORE["log"].clear()
                LOG_FILE.unlink(missing_ok=True)
            st.toast("Log deleted.")
            st.rerun()


{"Home": page_home, "Detect": page_detect, "Batch inspection": page_batch, "Model & choice": page_model,
 "Admin": page_admin}[page]()

st.markdown(f'<div class="ra-footer">Road Anomaly Detector · Deep Learning project (5BI2) · model {html.escape(det.name)} · '
            "data: RoadAnomaly (Lis et al., ICCV 2019), SegmentMeIfYouCan (Chan et al., NeurIPS 2021), Wikimedia "
            "Commons photos under CC licences · decision-support prototype</div>", unsafe_allow_html=True)
