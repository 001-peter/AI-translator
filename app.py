import os
import sys
import time
import platform
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Force system standard streams to use UTF-8 to prevent charmap UnicodeEncodeErrors in Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Global Constants
MODEL_NAME = "facebook/nllb-200-distilled-600M"
CACHE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

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

# Define Translation Engine
class TranslationEngine:
    def __init__(self, model_name=MODEL_NAME, use_quantization=False):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        local_model_path = os.path.join(CACHE_DIR, "local_nllb")
        if os.path.exists(local_model_path):
            load_path = local_model_path
            local_files_only = True
            print(f"Loading model/tokenizer from local path: {load_path}")
        else:
            load_path = self.model_name
            local_files_only = False
            print(f"Loading model/tokenizer for {self.model_name} from cache/web...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path, 
            cache_dir=CACHE_DIR if load_path == self.model_name else None,
            local_files_only=local_files_only
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            load_path,
            cache_dir=CACHE_DIR if load_path == self.model_name else None,
            local_files_only=local_files_only
        )
        
        if self.use_quantization:
            print("Applying PyTorch dynamic 8-bit quantization targeting linear layers...")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        if not text or not text.strip():
            return {"translation": "", "latency": 0.0, "character_rate": 0.0}
            
        start_time = time.time()
        try:
            self.tokenizer.src_lang = src_lang
            inputs = self.tokenizer(text, return_tensors="pt")
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=256
                )
            
            translated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"[TRANSLATION FAILURE] {e}")
            raise e
        
        latency = time.time() - start_time
        num_chars = len(text)
        char_rate = num_chars / latency if latency > 0 else 0.0
        
        return {
            "translation": translated_text,
            "latency": latency,
            "character_rate": char_rate
        }

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

# 3. Initialize Cache & Session States
if "translation_cache" not in st.session_state:
    st.session_state.translation_cache = {}
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

# Cache model resource per quantization option to avoid reloading unless settings change
@st.cache_resource(show_spinner=False)
def get_engine(use_quantization: bool = False) -> TranslationEngine:
    return TranslationEngine(use_quantization=use_quantization)

# Load engine with a friendly spinner
engine_ready = False
try:
    with st.spinner("Loading/Initializing offline translation model (this may take up to a minute on first run)..."):
        engine = get_engine(use_quantization=False) # Preload unquantized default
        engine_ready = True
except Exception as e:
    st.error(f"Failed to load translation engine: {e}")

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
cached_queries_count = len(st.session_state.translation_cache)
st.sidebar.write(f"Cached translations: **{cached_queries_count}**")

if st.sidebar.button("Clear Translation Cache"):
    st.session_state.translation_cache.clear()
    st.sidebar.success("Cache cleared!")
    time.sleep(0.5)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    ### ℹ️ Environment Details
    - **Host OS**: {platform.system()}
    - **Engine Status**: {"Ready" if engine_ready else "Offline / Loading"}
    - **Storage Location**: 
      `{CACHE_DIR}`
    """
)

# 7. Main Panel Header
st.markdown("<h1>🔮 AI Powered Translator</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #a5b4fc; margin-bottom: 20px;'>High-fidelity, privacy-centric Neural Machine Translation running 100% locally.</p>", unsafe_allow_html=True)

# 10. Translation Interface Elements
col_lang1, col_lang2 = st.columns(2)

lang_names = list(SUPPORTED_LANGUAGES.keys())

with col_lang1:
    src_lang_name = st.selectbox(
        "Translate From",
        options=lang_names,
        index=st.session_state.src_lang_idx,
        key="src_lang_select"
    )
    st.session_state.src_lang_idx = lang_names.index(src_lang_name)
        
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
    
    # Trigger translation via internal cache or in-process engine
    if translate_pressed and input_text.strip():
        text_clean = input_text.strip()
        # Check Cache
        cache_key = (text_clean, src_code, tgt_code, use_quantization)
        
        if cache_key in st.session_state.translation_cache:
            cached_result = st.session_state.translation_cache[cache_key]
            st.session_state.translated_text = cached_result["translation"]
            st.session_state.latency = cached_result["latency"]
            st.session_state.char_rate = cached_result["character_rate"]
            st.session_state.is_cached = True
            st.session_state.show_metrics = True
        else:
            with st.spinner("Translating text locally..."):
                try:
                    # Dynamically get engine (this will reload/quantize if setting changes)
                    engine = get_engine(use_quantization=use_quantization)
                    result = engine.translate(text_clean, src_code, tgt_code)
                    
                    # Store in cache
                    st.session_state.translation_cache[cache_key] = result
                    
                    st.session_state.translated_text = result["translation"]
                    st.session_state.latency = result["latency"]
                    st.session_state.char_rate = result["character_rate"]
                    st.session_state.is_cached = False
                    st.session_state.show_metrics = True
                except Exception as e:
                    st.error(f"Translation failed: {e}")
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
