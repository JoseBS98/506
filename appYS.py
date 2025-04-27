# appYS.py
import streamlit as st
import pandas as pd, numpy as np, joblib, io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestRegressor

# ── file locations (all paths are relative to this script) ─────────
ROOT        = Path(__file__).parent
MODEL_PATH  = ROOT / "yield_strength_rf.pkl"
DATA_PATH   = ROOT / "Parameters_simulation.csv"   # change if needed
PRINTER_IMG = ROOT / "printer.png"                  # M2 picture

# ── load existing model or train a new one on the fly ──────────────
def get_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        if not DATA_PATH.exists():
            st.error(f"Training file '{DATA_PATH.name}' not found.")
            st.stop()
        df = pd.read_csv(DATA_PATH)
        X  = df[['hatch_spacing','laser_beam_power','laser_beam_speed',
                 'laser_spot_size','scan_rotation','stripe_width']]
        y  = df['yield_strength']
        rf = RandomForestRegressor(n_estimators=500, max_depth=15,
                                   random_state=42)
        rf.fit(X, y)
        joblib.dump(rf, MODEL_PATH)
        return rf

model = get_model()

# ── basic Streamlit page setup ─────────────────────────────────────
st.set_page_config(page_title="Yield-Strength Calculator",
                   page_icon="🖨️", layout="wide")

st.markdown(
    """
    <style>
        body { background: #000; }
        h1, h3, label { color: #fff !important; }
        .stButton > button { background: #444; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center;'>Predicted Yield Strength</h1>",
            unsafe_allow_html=True)

# ── two-column layout ──────────────────────────────────────────────
left, right = st.columns(2, gap="large")

# -------- left: printer image placeholder -------------------------
if not PRINTER_IMG.exists():
    st.error(f"Image '{PRINTER_IMG.name}' missing.")
    st.stop()
base_img = Image.open(PRINTER_IMG).convert("RGBA")
img_slot = left.empty()
# use_container_width instead of the deprecated use_column_width
img_slot.image(base_img, use_container_width=True)

# -------- right: parameter inputs ---------------------------------
with right:
    st.markdown("### 3-D Printer Yield-Strength Calculator")
    st.markdown("Set the operating parameters then click **Predict**.")

    hatch   = st.number_input("Hatch spacing (µm)",   70.0,110.0, 90.0, step=0.1)
    power   = st.number_input("Laser beam power (W)", 300.0,450.0,360.0, step=0.1)
    speed   = st.number_input("Laser beam speed (mm/s)",
                              1200.0,1600.0,1350.0, step=1.0)
    spot    = st.number_input("Laser spot size (µm)",110.0,150.0,127.0, step=0.1)
    rot     = st.number_input("Scan rotation (deg)",    0.0, 90.0, 65.0, step=0.5)
    stripe  = st.number_input("Stripe width (mm)",      5.0, 20.0, 10.0, step=0.1)

    if st.button("Predict Yield Strength"):
        # 1) predict
        X_new  = np.array([[hatch, power, speed, spot, rot, stripe]])
        y_pred = model.predict(X_new)[0]
        text   = f"{y_pred:.1f} MPa"

        # 2) draw on a copy of the printer image
        img  = base_img.copy().convert("RGB")
        draw = ImageDraw.Draw(img)

        # 3) choose a large font
        try:
            font = ImageFont.truetype("arial.ttf", 10000)
        except:
            font = ImageFont.load_default()

        # 4) compute text size via textbbox (Pillow ≥10)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except AttributeError:
            # fallback for older Pillow versions
            w, h = font.getsize(text)

        # 5) center the text
        W, H = img.size
        x, y = (W - w)//2, (H - h)//2

        # 6) draw shadow + main text
        draw.text((x+2, y+2), text, fill="black",    font=font)
        draw.text((x,   y),   text, fill="#ffd600", font=font)

        # 7) send it back at a fixed width so it’s not auto-scaled
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_slot.image(buf.getvalue(), width=900)
