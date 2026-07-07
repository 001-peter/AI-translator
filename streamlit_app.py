import os
import sys
import time

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Force UTF-8 output for Windows console compatibility.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

CACHE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SUPPORTED_LANGUAGES = {
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
    "Yoruba (Èdè Yorùbá)": "yor_Latn",
    "Igbo (Asụsụ Igbo)": "ibo_Latn",
    "Hausa (Harshen Hausa)": "hau_Latn",
    "Nigerian Pidgin": "pcm_Latn",
}


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
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            load_path,
            cache_dir=CACHE_DIR if load_path == self.model_name else None,
            local_files_only=local_files_only,
        )

        if self.use_quantization:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8,
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
                    max_length=256,
                )

            translated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        except Exception as exc:
            print(f"[TRANSLATION FAILURE] {exc}")
            raise exc

        latency = time.time() - start_time
        num_chars = len(text)
        char_rate = num_chars / latency if latency > 0 else 0.0

        return {
            "translation": translated_text,
            "latency": latency,
            "character_rate": char_rate,
        }


@st.cache_resource(show_spinner=False)
def get_engine(use_quantization: bool = False) -> TranslationEngine:
    return TranslationEngine(use_quantization=use_quantization)


def translate_text(text: str, src_lang: str, tgt_lang: str, use_quantization: bool = False) -> dict:
    engine = get_engine(use_quantization=use_quantization)
    return engine.translate(text, src_lang, tgt_lang)


st.set_page_config(
    page_title="AI Powered Translator",
    page_icon="🔮",
    layout="wide",
)

st.title("AI Powered Translator")
st.markdown(
    "High-fidelity, privacy-centric Neural Machine Translation running 100% locally."
)

if "translation_result" not in st.session_state:
    st.session_state.translation_result = ""
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {}

with st.sidebar:
    st.header("Settings")
    use_quantization = st.checkbox("Enable dynamic quantization", value=False)
    st.caption("The model is loaded once and reused across requests.")

lang_names = list(SUPPORTED_LANGUAGES.keys())
col1, col2 = st.columns(2)
with col1:
    src_lang_name = st.selectbox("Translate from", options=lang_names, index=0)
with col2:
    tgt_lang_name = st.selectbox("Translate to", options=lang_names, index=1)

src_code = SUPPORTED_LANGUAGES[src_lang_name]
tgt_code = SUPPORTED_LANGUAGES[tgt_lang_name]

input_text = st.text_area(
    "Input text",
    height=220,
    placeholder="Enter text to translate here...",
)

if st.button("Translate", type="primary"):
    if not input_text.strip():
        st.warning("Please enter text to translate.")
    else:
        with st.spinner("Translating..."):
            result = translate_text(
                input_text.strip(),
                src_code,
                tgt_code,
                use_quantization=use_quantization,
            )
        st.session_state.translation_result = result["translation"]
        st.session_state.last_metrics = {
            "latency": result.get("latency", 0.0),
            "char_rate": result.get("character_rate", 0.0),
        }

st.subheader("Translation")
st.text_area(
    "Result",
    value=st.session_state.translation_result,
    height=220,
    disabled=False,
)

if st.session_state.last_metrics:
    st.caption(
        f"Latency: {st.session_state.last_metrics['latency']:.3f}s | "
        f"Speed: {st.session_state.last_metrics['char_rate']:.1f} chars/sec"
    )
