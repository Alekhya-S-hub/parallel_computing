# ♻️ Recyclable Image Training: Serial vs Parallel Demo (CPU)

This simple Streamlit app demonstrates **data parallelism** in distributed machine learning using a small subset of a recyclable image dataset.  
It compares **serial** (single-worker) and **parallel** (multi-worker) data loading in PyTorch on CPU.

---

## 🧱 Project Structure


> Each subfolder should contain about **200 images** (JPG/PNG) for a quick demo.

---
```bash
conda create -n pytorch311 python=3.11.7 -y
conda activate pytorch311
```
## ⚙️ Local Setup Instructions


**Windows:**
```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
