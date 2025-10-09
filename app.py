import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os, time
from pathlib import Path

# --- Setup ---
st.set_page_config(page_title="Parallel Inference Demo", layout="wide")
st.title("🧠 Distributed Parallel Image Classification Demo")
st.write("Compare serial vs. parallel GPU inference using a pretrained ResNet18 model.")

# --- Paths ---
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Load Model ---
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
st.success(f"Model loaded successfully on {device} ✅")

# --- Define preprocessing ---
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CLASSES = ["glass", "metal", "paper", "plastic"]

# --- File Upload ---
uploaded_files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

# --- Session variables for timing ---
if "serial_time" not in st.session_state:
    st.session_state.serial_time = None
if "parallel_time" not in st.session_state:
    st.session_state.parallel_time = None

if uploaded_files:
    st.write(f"✅ {len(uploaded_files)} images uploaded.")
    image_paths = []
    for file in uploaded_files:
        img_path = UPLOAD_DIR / file.name
        with open(img_path, "wb") as f:
            f.write(file.getbuffer())
        image_paths.append(img_path)

    cols = st.columns(min(4, len(uploaded_files)))
    for idx, img_path in enumerate(image_paths[:8]):
        img = Image.open(img_path)
        cols[idx % 4].image(img, caption=img_path.name, width=150)

    st.divider()

    # --- Serial Inference ---
    if st.button("▶ Run Serial Inference"):
        progress = st.progress(0)
        start = time.time()
        serial_results = []

        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_t)
                _, pred = torch.max(outputs, 1)
                serial_results.append(CLASSES[pred.item()])
            progress.progress((i + 1) / len(image_paths))

        end = time.time()
        st.session_state.serial_time = end - start
        serial_throughput = len(image_paths) / st.session_state.serial_time

        st.subheader("Serial Inference Results:")
        for i, (path, pred) in enumerate(zip(image_paths, serial_results)):
            st.write(f"**{path.name} → {pred}**")

        st.info(f"🕒 Serial inference time: {st.session_state.serial_time:.2f}s | "
                f"Throughput: {serial_throughput:.2f} images/sec")

    # --- Parallel Inference ---
    if st.button("⚡ Run Parallel Inference"):
        batch_size = st.slider("Select batch size", 8, 128, 32, 8)
        progress = st.progress(0)
        start = time.time()

        images = []
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert("RGB")
            img_t = transform(img)
            images.append(img_t)
            progress.progress((i + 1) / len(image_paths))

        batch = torch.stack(images).to(device)
        with torch.no_grad():
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
        end = time.time()

        st.session_state.parallel_time = end - start
        parallel_throughput = len(image_paths) / st.session_state.parallel_time

        st.subheader("Parallel Inference Results:")
        for i, (path, pred) in enumerate(zip(image_paths, preds)):
            st.write(f"**{path.name} → {CLASSES[pred.item()]}**")

        st.success(f"⚡ Parallel inference time: {st.session_state.parallel_time:.2f}s | "
                   f"Throughput: {parallel_throughput:.2f} images/sec")

        # --- Comparison Chart ---
        if st.session_state.serial_time:
            import pandas as pd
            import altair as alt

            df = pd.DataFrame({
                "Mode": ["Serial", "Parallel"],
                "Throughput (images/sec)": [
                    len(image_paths) / st.session_state.serial_time,
                    parallel_throughput
                ]
            })

            chart = (
                alt.Chart(df)
                .mark_bar(size=50)
                .encode(
                    x=alt.X("Mode", sort=None),
                    y="Throughput (images/sec)",
                    color="Mode"
                )
                .properties(title="Throughput Comparison: Serial vs Parallel")
            )

            st.altair_chart(chart, use_container_width=True)

            # --- Speed-up Display ---
            speedup = st.session_state.serial_time / st.session_state.parallel_time
            st.write(f"💡 **Speed-up:** {speedup:.2f}× faster (Parallel vs Serial)")
