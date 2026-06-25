import os
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define a local model cache directory to keep the project self-contained
CACHE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# FLORES-200 language codes for NLLB-200
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

class TranslationEngine:
    def __init__(self, model_name=MODEL_NAME, use_quantization=True):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        local_path = os.path.join(CACHE_DIR, "local_nllb")
        local_files_only = False
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "pytorch_model.bin")):
            load_path = local_path
            local_files_only = True
            print(f"Loading tokenizer and model from local path: {load_path}")
        else:
            load_path = self.model_name
            print(f"Loading tokenizer and model for {self.model_name} from cache/web...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path, 
            cache_dir=CACHE_DIR if load_path == self.model_name else None,
            local_files_only=local_files_only
        )
        raw_model = AutoModelForSeq2SeqLM.from_pretrained(
            load_path, 
            cache_dir=CACHE_DIR if load_path == self.model_name else None,
            local_files_only=local_files_only
        )
        
        if self.use_quantization:
            print("Applying dynamic quantization to model (CPU)...")
            start_time = time.time()
            # Quantize PyTorch linear layers dynamically to 8-bit integers
            self.model = torch.quantization.quantize_dynamic(
                raw_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print(f"Quantization completed in {time.time() - start_time:.2f}s.")
        else:
            self.model = raw_model
            
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        """
        Translates text from src_lang to tgt_lang.
        Returns a dictionary containing the translated text and performance metrics.
        """
        if not text or not text.strip():
            return {"translation": "", "latency": 0.0, "character_rate": 0.0}
            
        start_time = time.time()
        
        # Set source language in tokenizer
        self.tokenizer.src_lang = src_lang
        
        # Tokenize inputs
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Get target language token ID
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id
            )
            
        # Decode translation
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        latency = time.time() - start_time
        num_chars = len(text)
        char_rate = num_chars / latency if latency > 0 else 0.0
        
        return {
            "translation": translated_text,
            "latency": latency,
            "character_rate": char_rate
        }
