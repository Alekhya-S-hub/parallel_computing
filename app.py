import io
import time
from PIL import Image
import streamlit as st
import torch
from torchvision import models, transforms
import pandas as pd
import altair as alt

# ===========================
# Streamlit Page Config
# ===========================
st.set_page_config(page_title="Parallel Inference Demo", layout="wide")
st.title("🧠 Distributed Parallel Image Classification Demo")
st.write("Compare **serial vs. parallel GPU inference** using a pretrained ResNet18 model.")

# ===========================
# Load Model
# ===========================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load("recycle_resnet18.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()
st.success(f"Model loaded successfully on **{device}** ✅")

# ===========================
# Image Transform
# ===========================
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CLASSES = ["glass", "metal", "paper", "plastic"]

# ===========================
# Session State Variables
# ===========================
if "serial_time" not in st.session_state:
    st.session_state.serial_time = None
if "parallel_time" not in st.session_state:
    st.session_state.parallel_time = None

# ===========================
# File Upload
# ===========================
uploaded_files = st.file_uploader(
    "📂 Upload multiple images",
    accept_multiple_files=True,
    type=["jpg", "png", "jpeg"]
)

if uploaded_files:
    st.write(f"✅ **{len(uploaded_files)} images uploaded.**")

    cols = st.columns(min(4, len(uploaded_files)))
    for idx, file in enumerate(uploaded_files[:8]):
        img = Image.open(file)
        cols[idx % 4].image(img, caption=file.name, width=150)

    st.divider()

    # ===========================
    # Serial Inference
    # ===========================
    if st.button("▶ Run Serial Inference"):
        progress = st.progress(0)
        start = time.time()
        serial_results = []

        for i, file in enumerate(uploaded_files):
            img = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_t)
                _, pred = torch.max(outputs, 1)
                serial_results.append(CLASSES[pred.item()])
            progress.progress((i + 1) / len(uploaded_files))

        end = time.time()
        st.session_state.serial_time = end - start
        serial_throughput = len(uploaded_files) / st.session_state.serial_time

        st.subheader("Serial Inference Results")
        for f, pred in zip(uploaded_files, serial_results):
            st.write(f"**{f.name} → {pred}**")

        st.info(
            f"🕒 Serial inference time: **{st.session_state.serial_time:.2f}s** | "
            f"Throughput: **{serial_throughput:.2f} images/sec**"
        )

    # ===========================
    # Parallel Inference
    # ===========================
    # --- Batch size slider (set before running) ---
    if "batch_size" not in st.session_state:
        st.session_state.batch_size = 32

    st.session_state.batch_size = st.slider(
        "Select batch size for Parallel Inference",
        min_value=8,
        max_value=128,
        step=8,
        value=st.session_state.batch_size
    )

    # --- Parallel Inference Button ---
    if st.button("⚡ Run Parallel Inference"):
        batch_size = st.session_state.batch_size
        progress = st.progress(0)
        start = time.time()

        images = []
        for i, file in enumerate(uploaded_files):
            img = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
            img_t = transform(img)
            images.append(img_t)
            progress.progress((i + 1) / len(uploaded_files))

        # Combine into single batch tensor
        batch = torch.stack(images).to(device)
        with torch.no_grad():
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
        end = time.time()

        st.session_state.parallel_time = end - start
        parallel_throughput = len(uploaded_files) / st.session_state.parallel_time

        st.subheader("Parallel Inference Results")
        for f, pred in zip(uploaded_files, preds):
            st.write(f"**{f.name} → {CLASSES[pred.item()]}**")

        st.success(
            f"⚡ Parallel inference time: **{st.session_state.parallel_time:.2f}s** | "
            f"Throughput: **{parallel_throughput:.2f} images/sec**"
        )

        # ===========================
        # Comparison Chart
        # ===========================
        if st.session_state.serial_time:
            df = pd.DataFrame({
                "Mode": ["Serial", "Parallel"],
                "Throughput (images/sec)": [
                    len(uploaded_files) / st.session_state.serial_time,
                    parallel_throughput
                ]
            })

            chart = (
                alt.Chart(df)
                .mark_bar(size=60)
                .encode(
                    x=alt.X("Mode", sort=None),
                    y="Throughput (images/sec)",
                    color="Mode"
                )
                .properties(title="Throughput Comparison: Serial vs Parallel")
            )
            st.altair_chart(chart, use_container_width=True)

            speedup = st.session_state.serial_time / st.session_state.parallel_time
            st.markdown(f"💡 **Speed-up:** `{speedup:.2f}× faster (Parallel vs Serial)`")
else:
    st.info("👆 Please upload a few images to begin.")
