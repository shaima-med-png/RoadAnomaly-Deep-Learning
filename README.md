# Road anomaly detector: Streamlit version

Streamlit web application for the best model of `notebooks/RoadAnomaly_complete.ipynb`, with a permanent public link
on **Streamlit Community Cloud** (free). Everything is done in the browser: nothing to install on your computer.

## Use cases

| # | Use case | Where in the app |
|---|---|---|
| UC1 | Analyse a road image: red mask + alert (number of obstacles, area, position) | tab **Analyse an image** |
| UC2 | Adjust the sensitivity: 3 presets + threshold slider | sidebar |
| UC3 | See the anomaly score map | tab 1, next to the mask |
| UC4 | Explain the detection with Grad-CAM | sidebar checkbox, tab 1 |
| UC5 | Inspect a batch of images, CSV report | tab **Batch inspection** |
| UC6 | Export the alert as JSON for another system | tab 1, *Alert details and export* |
| UC7 | Read the model card: test metrics and limits | tab **Model card** |

## Files

| File | Role |
|---|---|
| `streamlit_app.py` | the interface (Streamlit) |
| `roadanomaly_core.py` | model code and inference pipeline, without interface (same code as the notebook) |
| `requirements.txt` | packages; the first line makes Streamlit Cloud install the light CPU version of PyTorch |
| `examples/` | 4 test images and their credits (`CREDITS.txt`) |
| `roadanomaly_model.pt` | **not in the repository**: the model exported by section 7.1 of the notebook (about 57 MB), published as a GitHub release file |

## Deploy on Streamlit Community Cloud (about 15 minutes)

You need `roadanomaly_model.pt` (section 7.1 of the notebook downloads it automatically on Colab) and a free
GitHub account.

**1. Create the repository**
1. On github.com: **New repository**, name `road-anomaly-detector`, **Public**, tick *Add a README file*, **Create**.
2. **Add file → Upload files**: drag `streamlit_app.py`, `roadanomaly_core.py`, `requirements.txt` and the
   `examples` folder, then **Commit changes**.

**2. Publish the model as a release file**
The GitHub web page refuses files over 25 MB, but a *release* accepts large files.
1. In the repository: **Releases → Create a new release**. Tag: `v1`, title: `Model v1`.
2. Drag `roadanomaly_model.pt` into *Attach binaries*, then **Publish release**.
3. Right-click the file name in the release → **Copy link**. It looks like
   `https://github.com/<your-user>/road-anomaly-detector/releases/download/v1/roadanomaly_model.pt`.

**3. Create the app**
1. Go to **share.streamlit.io** and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**: repository `<your-user>/road-anomaly-detector`, branch `main`,
   main file `streamlit_app.py`. Choose an app URL, for example `road-anomaly-detector`.
3. **Advanced settings**: Python 3.11, and in **Secrets** paste (with your link):
   ```toml
   MODEL_URL = "https://github.com/<your-user>/road-anomaly-detector/releases/download/v1/roadanomaly_model.pt"
   ```
4. **Deploy**. The first start installs PyTorch and downloads the model (a few minutes). The app is then at
   `https://road-anomaly-detector.streamlit.app` (or the name you chose).

The free app goes to sleep after a few days without visits; opening the link wakes it up (about a minute).

## Run on a computer with Python (optional)

Put `roadanomaly_model.pt` in this folder, then:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Live API (UC6, optional)

Streamlit makes interactive pages, not APIs: in the app, UC6 is the JSON export of the alert. A program that needs a
live answer can use the same `Detector` class in a small FastAPI service:
```python
from fastapi import FastAPI, UploadFile
from PIL import Image
from roadanomaly_core import Detector, alert_from

det = Detector("roadanomaly_model.pt")
api = FastAPI()

@api.post("/detect")
async def detect(file: UploadFile, threshold: float = 0.47):
    rgb = Image.open(file.file).convert("RGB")
    prob, latency = det.score_map(rgb)
    return {"latency (ms)": round(latency, 1), **alert_from(prob, threshold)[1]}
```

## Limits

Trained on 60 daytime web images: not validated for night, rain, other cameras or video. People and ordinary
vehicles can trigger alerts. Recall is about 0.75. Decision-support prototype, not a safety-certified component.
