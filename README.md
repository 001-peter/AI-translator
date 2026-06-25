# 🔮 Antigravity NMT Studio (NLLB-200)

A local, privacy-centric Neural Machine Translation (NMT) web application built using Python, Hugging Face Transformers, PyTorch, and Streamlit. It uses the `facebook/nllb-200-distilled-600M` model to support direct translations between 200+ languages offline, including major international languages and Nigerian indigenous languages (Yoruba, Igbo, Hausa, and Nigerian Pidgin).

To keep the application responsive and lightweight on CPU, **PyTorch Dynamic 8-bit Quantization** is applied to the linear layers of the transformer, reducing the runtime RAM footprint by ~60% (~700 MB RAM) and accelerating inference speed.

---

## 📁 Project Structure

- **`app.py`**: The Streamlit frontend application.
- **`translator.py`**: The backend machine learning logic (model loading, dynamic quantization, and inference).
- **`styles.css`**: Premium glassmorphic interface style customizations.
- **`test_translator.py`**: Integration test to verify that loading, quantization, and translation run successfully between English and Yoruba.
- **`requirements.txt`**: Specifies python library dependencies.
- **`python-portable/`**: Bundled portable Python environment.

---

## 🚀 How to Run

Since a portable Python distribution is pre-extracted in the project workspace, you can execute the project directly without installing Python system-wide.

### 1. Verify Dependencies
The dependencies are already installed in the portable environment. If you need to reinstall them:
```powershell
.\python-portable\tools\python.exe -m pip install -r requirements.txt
```

### 2. Run Automated Verification Test
To verify the translation model loads and translates correctly:
```powershell
.\python-portable\tools\python.exe test_translator.py
```
*Note: On the first run, this will download the 2.4 GB model file from Hugging Face and cache it locally in the `model_cache/` directory. Subsequent runs will be fully offline and instant.*

### 3. Start the Web Application
Launch the Streamlit server:
```powershell
.\python-portable\tools\python.exe -m streamlit run app.py
```

After launching, open your browser and navigate to the local address provided in the terminal (typically `http://localhost:8501`).

---

## ⚙️ Advanced Customizations

You can modify settings from the web interface sidebar:
- **Enable/Disable Dynamic Quantization**: Dynamic Quantization reduces memory usage and speeds up CPU execution. You can disable it to test translation quality differences (differences are typically negligible).
- **Cache Management**: The application caches translations during your session to provide instant results for repeat queries. You can clear this cache with the "Clear Translation Cache" button in the sidebar.
