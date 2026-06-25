import os
import sys
import time

# Force system standard streams to use UTF-8 to prevent charmap UnicodeEncodeErrors in Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import streamlit as st
from translator import TranslationEngine, SUPPORTED_LANGUAGES, CACHE_DIR, MODEL_NAME

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

# 3. Model Caching Utility
@st.cache_resource(show_spinner=False)
def load_translation_engine(model_name, use_quantization):
    return TranslationEngine(model_name=model_name, use_quantization=use_quantization)

# 4. Translation Cache to prevent redundant runs
if "translation_cache" not in st.session_state:
    st.session_state.translation_cache = {}

def get_cached_translation(text, src_code, tgt_code, engine, use_quantize):
    cache_key = (text.strip(), src_code, tgt_code, use_quantize)
    if cache_key in st.session_state.translation_cache:
        return st.session_state.translation_cache[cache_key], True
    
    # Run translation
    result = engine.translate(text, src_code, tgt_code)
    st.session_state.translation_cache[cache_key] = result
    return result, False

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
    value=True,
    help="Reduces RAM usage by ~60% (~700MB runtime) and speeds up CPU inference. Recommended."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗄️ Cache Management")
cached_queries_count = len(st.session_state.translation_cache)
st.sidebar.write(f"Cached translations: **{cached_queries_count}**")

if st.sidebar.button("Clear Translation Cache"):
    st.session_state.translation_cache.clear()
    st.sidebar.success("Cache cleared!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    ### ℹ️ Environment Details
    - **Host OS**: Windows
    - **Engine Status**: Ready
    - **Storage Location**: 
      `{CACHE_DIR}`
    """
)

# 7. Main Panel Header
st.markdown("<h1>🔮 AI Powered Translator</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #a5b4fc; margin-bottom: 20px;'>High-fidelity, privacy-centric Neural Machine Translation running 100% locally.</p>", unsafe_allow_html=True)



# 9. Load the Translation Engine
try:
    with st.spinner("Initializing Translation Model (loading files and applying optimizations... this may take a moment)..."):
        engine = load_translation_engine(selected_model, use_quantization)
    model_loaded = True
except Exception as e:
    st.error(f"❌ Failed to load the translation model: {e}")
    model_loaded = False

# 10. Translation Interface Elements
if model_loaded:
    # Set up columns for Source/Target language selection
    col_lang1, col_swap, col_lang2 = st.columns([9, 2, 9])
    
    lang_names = list(SUPPORTED_LANGUAGES.keys())
    
    with col_lang1:
        src_lang_name = st.selectbox(
            "Translate From",
            options=lang_names,
            index=st.session_state.src_lang_idx,
            key="src_lang_select"
        )
        # Update session state index
        st.session_state.src_lang_idx = lang_names.index(src_lang_name)
        
    with col_swap:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("↔ Swap", help="Swap source and target languages"):
            # Swap indices
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
        # Update session state index
        st.session_state.tgt_lang_idx = lang_names.index(tgt_lang_name)

    # Get raw codes
    src_code = SUPPORTED_LANGUAGES[src_lang_name]
    tgt_code = SUPPORTED_LANGUAGES[tgt_lang_name]

    # Set up Columns for Text Input and Output
    col_txt1, col_txt2 = st.columns(2)
    
    with col_txt1:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.markdown(f"### Source Text ({src_lang_name})")
        
        # User input text area
        input_placeholder = "Enter text to translate here..."
        input_text = st.text_area(
            "Source Input",
            height=250,
            placeholder=input_placeholder,
            label_visibility="collapsed",
            key="input_text"
        )
        
        # Counters
        char_count = len(input_text)
        word_count = len(input_text.split()) if input_text else 0
        st.markdown(
            f"<div style='text-align: right; font-size: 13px; color: #94a3b8; margin-top: 5px;'>"
            f"Characters: {char_count} | Words: {word_count}</div>",
            unsafe_allow_html=True
        )
        
        # Action Buttons
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
        
        # Perform Translation
        if translate_pressed and input_text.strip():
            print(f"DEBUG: Translate clicked! input_text={repr(input_text[:50])}..., src='{src_code}', tgt='{tgt_code}'")
            with st.spinner("Translating..."):
                try:
                    result, is_cached = get_cached_translation(
                        input_text, src_code, tgt_code, engine, use_quantization
                    )
                    st.session_state.translated_text = result["translation"]
                    st.session_state.latency = result["latency"]
                    st.session_state.char_rate = result["character_rate"]
                    st.session_state.is_cached = is_cached
                    st.session_state.show_metrics = True
                    print(f"DEBUG: Translation success! result={repr(st.session_state.translated_text[:50])}...")
                except Exception as e:
                    st.error(f"Translation Error: {e}")
                    st.session_state.translated_text = f"Error: {e}"
                    st.session_state.show_metrics = False
                    print(f"DEBUG: Translation error: {repr(e)}")
        elif input_text.strip() == "":
            st.session_state.translated_text = ""
            st.session_state.show_metrics = False
            
        # Output text area (rendered as read-only / styled)
        st.text_area(
            "Target Output",
            value=st.session_state.translated_text,
            height=250,
            disabled=False,
            label_visibility="collapsed"
        )
        
        # Counters
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
