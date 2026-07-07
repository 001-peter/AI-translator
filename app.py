import os
import sys
import time
import requests
import streamlit as st

# Force system standard streams to use UTF-8 to prevent charmap UnicodeEncodeErrors in Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Global Constants copied from translator to bypass PyTorch/Transformers import time
MODEL_NAME = "facebook/nllb-200-distilled-600M"
CACHE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "model_cache")
SUPPORTED_LANGUAGES = {
    # International Languages
    "English": "eng_Latn",
    "Spanish (Español)": "spa_Latn",
    "French (Français)": "fra_Latn",
    "German (Deutsch)": "deu_Latn",
    "Chinese (Simplified)": "zho_Hans",
    "Japanese (日本語)": "jpn_Jpan",
    "Italian (Italiano)": "ita_Latn",
    "Portuguese (Português)": "por_Latn",
    "Russian (Русский)": "rus_Latn",
    "Arabic (العربية)": "ara_Arab",
    "Hindi (हिन्दी)": "hin_Deva",
    "Korean (한국어)": "kor_Hang",
    "Dutch (Nederlands)": "nld_Latn",
    "Turkish (Türkçe)": "tur_Latn",
    "Swedish (Svenska)": "swe_Latn",
    "Polish (Polski)": "pol_Latn",
    "Vietnamese (Tiếng Việt)": "vie_Latn",
    "Indonesian (Bahasa Indonesia)": "ind_Latn",
    
    # Nigerian Indigenous Languages
    "Yoruba (Èdè Yorùbá)": "yor_Latn",
    "Igbo (Asụsụ Igbo)": "ibo_Latn",
    "Hausa (Harshen Hausa)": "hau_Latn",
    "Nigerian Pidgin": "pcm_Latn"
}

API_URL = "http://127.0.0.1:8000"

# 1. Page Configuration
st.set_page_config(
    page_title="AI Powered Translator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom CSS Stylesheet
def inject_custom_styles():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom stylesheet styles.css not found.")

inject_custom_styles()

# Helper function to check translation server health
def check_backend_health():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=1.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

# Check backend server status
server_status = check_backend_health()

# 5. Initialize Session State for Language Selections & Outputs
if "src_lang_idx" not in st.session_state:
    st.session_state.src_lang_idx = 0  # English default
if "tgt_lang_idx" not in st.session_state:
    st.session_state.tgt_lang_idx = 1  # Spanish default
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "latency" not in st.session_state:
    st.session_state.latency = 0.0
if "char_rate" not in st.session_state:
    st.session_state.char_rate = 0.0
if "is_cached" not in st.session_state:
    st.session_state.is_cached = False
if "show_metrics" not in st.session_state:
    st.session_state.show_metrics = False

# 6. Sidebar Panel
st.sidebar.markdown("## ⚙️ Translation Studio Settings")
st.sidebar.markdown("Configure your offline translation engine preferences.")

selected_model = st.sidebar.selectbox(
    "Translation Model",
    options=[MODEL_NAME],
    help="High-quality distilled multilingual model supporting 200+ languages including Nigerian indigenous languages."
)

use_quantization = st.sidebar.checkbox(
    "Enable Dynamic Quantization",
    value=False,
    help="Note: Quantization is disabled by default due to stability issues on Windows CPU."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗄️ Cache Management")
cached_queries_count = server_status.get("cached_queries", 0) if server_status else 0
st.sidebar.write(f"Cached translations: **{cached_queries_count}**")

if st.sidebar.button("Clear Translation Cache"):
    if server_status:
        try:
            requests.post(f"{API_URL}/clear_cache", timeout=1.0)
            st.sidebar.success("Cache cleared!")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to clear cache: {e}")
    else:
        st.sidebar.warning("Server offline. Cannot clear cache.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    ### ℹ️ Environment Details
    - **Host OS**: Windows
    - **Engine Status**: {"Ready" if server_status and server_status.get("status") == "ready" else "Offline / Loading"}
    - **Storage Location**: 
      `{CACHE_DIR}`
    """
)

# 7. Main Panel Header
st.markdown("<h1>🔮 AI Powered Translator</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #a5b4fc; margin-bottom: 20px;'>High-fidelity, privacy-centric Neural Machine Translation running 100% locally.</p>", unsafe_allow_html=True)

# 8. Handling Backend Connection Errors
if not server_status:
    st.error("❌ Cannot connect to the background Translation Server!")
    st.warning("Ensure that the translation server is running on `http://127.0.0.1:8000`. Double-click `run_studio.bat` or run the command below to start it:")
    st.code(".\\python-portable\\tools\\python.exe server.py")
    st.stop()

# 10. Translation Interface Elements
col_lang1, col_swap, col_lang2 = st.columns([9, 2, 9])

lang_names = list(SUPPORTED_LANGUAGES.keys())

with col_lang1:
    src_lang_name = st.selectbox(
        "Translate From",
        options=lang_names,
        index=st.session_state.src_lang_idx,
        key="src_lang_select"
    )
    st.session_state.src_lang_idx = lang_names.index(src_lang_name)
    
with col_swap:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("↔ Swap", help="Swap source and target languages"):
        temp = st.session_state.src_lang_idx
        st.session_state.src_lang_idx = st.session_state.tgt_lang_idx
        st.session_state.tgt_lang_idx = temp
        st.rerun()
        
with col_lang2:
    tgt_lang_name = st.selectbox(
        "Translate To",
        options=lang_names,
        index=st.session_state.tgt_lang_idx,
        key="tgt_lang_select"
    )
    st.session_state.tgt_lang_idx = lang_names.index(tgt_lang_name)

# Get language codes
src_code = SUPPORTED_LANGUAGES[src_lang_name]
tgt_code = SUPPORTED_LANGUAGES[tgt_lang_name]

# Text areas
col_txt1, col_txt2 = st.columns(2)

with col_txt1:
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.markdown(f"### Source Text ({src_lang_name})")
    
    input_placeholder = "Enter text to translate here..."
    input_text = st.text_area(
        "Source Input",
        height=250,
        placeholder=input_placeholder,
        label_visibility="collapsed",
        key="input_text"
    )
    
    char_count = len(input_text)
    word_count = len(input_text.split()) if input_text else 0
    st.markdown(
        f"<div style='text-align: right; font-size: 13px; color: #94a3b8; margin-top: 5px;'>"
        f"Characters: {char_count} | Words: {word_count}</div>",
        unsafe_allow_html=True
    )
    
    # Action buttons
    def clear_text():
        st.session_state.input_text = ""
        st.session_state.translated_text = ""
        st.session_state.show_metrics = False

    col_act1, col_act2 = st.columns([1, 1])
    with col_act1:
        st.button("Clear", on_click=clear_text)
    with col_act2:
        translate_pressed = st.button("Translate", type="primary")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
with col_txt2:
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.markdown(f"### Translated Text ({tgt_lang_name})")
    
    # Trigger translation via API POST request
    if translate_pressed and input_text.strip():
        with st.spinner("Requesting translation from local backend..."):
            try:
                payload = {
                    "text": input_text,
                    "src_lang": src_code,
                    "tgt_lang": tgt_code,
                    "use_quantization": use_quantization
                }
                
                resp = requests.post(f"{API_URL}/translate", json=payload, timeout=60.0)
                
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.translated_text = data["translation"]
                    st.session_state.latency = data["latency"]
                    st.session_state.char_rate = data["character_rate"]
                    st.session_state.is_cached = data["is_cached"]
                    st.session_state.show_metrics = True
                else:
                    st.error(f"API Translation Error: {resp.text}")
                    st.session_state.translated_text = f"Error: API returned status {resp.status_code}"
                    st.session_state.show_metrics = False
            except Exception as e:
                st.error(f"Failed to connect to translation server: {e}")
                st.session_state.translated_text = f"Error: {e}"
                st.session_state.show_metrics = False
    elif input_text.strip() == "":
        st.session_state.translated_text = ""
        st.session_state.show_metrics = False
        
    st.text_area(
        "Target Output",
        value=st.session_state.translated_text,
        height=250,
        disabled=False,
        label_visibility="collapsed"
    )
    
    out_char_count = len(st.session_state.translated_text)
    out_word_count = len(st.session_state.translated_text.split()) if st.session_state.translated_text else 0
    st.markdown(
        f"<div style='text-align: right; font-size: 13px; color: #94a3b8; margin-top: 5px;'>"
        f"Characters: {out_char_count} | Words: {out_word_count}</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
# Render Execution Metrics
if st.session_state.show_metrics and st.session_state.translated_text and not st.session_state.translated_text.startswith("Error:"):
    st.markdown("### 📊 Performance Analytics")
    metrics_html = f"""
    <div style="display: flex; flex-wrap: wrap; margin-bottom: 20px;">
        <div class="metric-badge metric-latency">
            ⏱️ Latency: {st.session_state.latency:.3f} s
        </div>
        <div class="metric-badge metric-rate">
            ⚡ Speed: {st.session_state.char_rate:.1f} chars/sec
        </div>
        <div class="metric-badge metric-status">
            🛠️ Quantized: {"Yes (8-bit)" if use_quantization else "No (32-bit)"}
        </div>
        <div class="metric-badge metric-status" style="background-color: rgba(59, 130, 246, 0.15); color: #93c5fd; border-color: rgba(59, 130, 246, 0.25);">
            💾 Cache hit: {"Yes" if st.session_state.is_cached else "No"}
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)
