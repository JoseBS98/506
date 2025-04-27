# app.py  ──────────────────────────────────────────────────────────
import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# -----------------------------------------------------------------
# Load model & printer photo
# -----------------------------------------------------------------
model      = joblib.load("yield_strength_rf.pkl")
printer_im = Image.open("printer.png").convert("RGBA")   # 1 × file

# -----------------------------------------------------------------
# Page config / dark mode
# -----------------------------------------------------------------
st.set_page_config(page_title="Yield-Strength Calculator",
                   page_icon="🖨️",
                   layout="wide",
                   initial_sidebar_state="auto")
st.markdown(
    """
    <style>
        body { background-color:#000000; }
        h1   { color:#ffffff; text-align:center; }
        h3   { color:#ffffff; }
        label, input, div[data-baseweb="input"]  { color:#ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True)

st.markdown("## &nbsp;")          # small vertical offset
st.markdown("<h1>Predicted Yield Strength</h1>", unsafe_allow_html=True)

# -----------------------------------------------------------------
# Layout: 2 equal columns
# -----------------------------------------------------------------
col_left, col_right = st.columns(2, gap="large")

# ---------------- Left column: printer photo + prediction ----------
with col_left:
    st.markdown("###")  # spacing
    # placeholder for dynamic image
    img_slot = st.empty()

# ---------------- Right column: parameter inputs -------------------
with col_right:
    st.markdown("### 3-D Printer Yield-Strength Calculator")
    st.markdown("Set the operating parameters below to predict the expected "
                "**Yield Strength** (MPa) of the printed part.")

    hatch      = st.number_input("Hatch spacing (µm)",        70.0, 110.0, 90.0,  step=0.1)
    power      = st.number_input("Laser beam power (W)",     300.0, 450.0, 360.0, step=0.1)
    speed      = st.number_input("Laser beam speed (mm/s)", 1200.0,1600.0,1350.0, step=1.0)
    spot       = st.number_input("Laser spot size (µm)",     110.0, 150.0, 127.0, step=0.1)
    rotation   = st.number_input("Scan rotation (deg)",        0.0,  90.0,  65.0, step=0.5)
    stripe     = st.number_input("Stripe width (mm)",          5.0,  20.0,  10.0, step=0.1)

    if st.button("Predict Yield Strength"):
        # --------------------------------------
        # 1.  Predict
        # --------------------------------------
        X_new  = np.array([[hatch, power, speed, spot, rotation, stripe]])
        y_pred = model.predict(X_new)[0]

        # --------------------------------------
        # 2.  Draw red rectangle + text on image
        # --------------------------------------
        im = printer_im.copy()
        draw = ImageDraw.Draw(im, "RGBA")

        # Rectangle position (manually tuned for the M2 image):
        rect = (120, 205, 460, 275)          # (x0,y0,x1,y1)
        draw.rectangle(rect, fill=(255,0,0,200))   # semi-
