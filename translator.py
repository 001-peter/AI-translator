import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
    def __init__(self, model_name=MODEL_NAME, use_quantization=False):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        # We load the model from cache_dir/local_nllb
        local_model_path = os.path.join(CACHE_DIR, "local_nllb")
        if os.path.exists(local_model_path):
            load_path = local_model_path
            local_files_only = True
            print(f"Loading model/tokenizer from local path: {load_path}")
        else:
            load_path = self.model_name
            local_files_only = False
            print(f"Loading model/tokenizer for {self.model_name} from cache/web...")
        
        # Load Hugging Face tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path, 
            cache_dir=CACHE_DIR if load_path == self.model_name else None,
            local_files_only=local_files_only
        )
        
        # Load PyTorch Seq2Seq Model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            load_path,
            cache_dir=CACHE_DIR if load_path == self.model_name else None,
            local_files_only=local_files_only
        )
        
        # Apply dynamic quantization to linear layers if enabled
        if self.use_quantization:
            print("Applying PyTorch dynamic 8-bit quantization targeting linear layers...")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        """
        Translates text from src_lang to tgt_lang using PyTorch.
        Returns a dictionary containing the translated text and performance metrics.
        """
        if not text or not text.strip():
            return {"translation": "", "latency": 0.0, "character_rate": 0.0}
            
        start_time = time.time()
        
        try:
            # Set source language on tokenizer
            self.tokenizer.src_lang = src_lang
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # Map target language code to its token ID
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
            
            # Run inference without tracking gradients (faster and uses less memory)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=256
                )
            
            # Decode generated output tokens
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
