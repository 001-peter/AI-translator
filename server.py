import os
import sys
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Force system standard streams to use UTF-8 to prevent console issues
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from translator import TranslationEngine, MODEL_NAME

app = FastAPI(
    title="Antigravity NMT Server",
    description="Offline Neural Machine Translation Engine API"
)

# Translation engine instances cached by quantization state
engines = {}
# Global in-memory translation cache (keys are tuples: (text, src, tgt, quantization))
translation_cache = {}

def get_engine(use_quantization: bool) -> TranslationEngine:
    global engines
    key = "quantized" if use_quantization else "unquantized"
    if key not in engines:
        print(f"[API SERVER] Loading PyTorch TranslationEngine (quantized={use_quantization})...", flush=True)
        engines[key] = TranslationEngine(use_quantization=use_quantization)
        print(f"[API SERVER] PyTorch TranslationEngine (quantized={use_quantization}) loaded successfully.", flush=True)
    return engines[key]

@app.on_event("startup")
def startup_event():
    print("[API SERVER] Starting up...", flush=True)
    # Pre-load the default unquantized PyTorch engine to avoid first-request latency
    try:
        get_engine(use_quantization=False)
    except Exception as e:
        print(f"[API SERVER] Warning: Failed to pre-load PyTorch model on startup: {e}", flush=True)

class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    use_quantization: bool = False

@app.get("/health")
def health():
    return {
        "status": "ready" if "default" in engines else "loading",
        "engines_loaded": list(engines.keys()),
        "cached_queries": len(translation_cache)
    }

@app.post("/clear_cache")
def clear_cache():
    translation_cache.clear()
    return {"status": "success", "message": "Translation cache cleared"}

@app.post("/translate")
def translate(req: TranslationRequest):
    text_clean = req.text.strip()
    if not text_clean:
        return {
            "translation": "",
            "latency": 0.0,
            "character_rate": 0.0,
            "is_cached": False
        }

    # Check Cache
    cache_key = (text_clean, req.src_lang, req.tgt_lang, req.use_quantization)
    if cache_key in translation_cache:
        cached_result = translation_cache[cache_key]
        return {
            "translation": cached_result["translation"],
            "latency": cached_result["latency"],
            "character_rate": cached_result["character_rate"],
            "is_cached": True
        }

    # Perform Translation
    try:
        engine = get_engine(req.use_quantization)
        result = engine.translate(text_clean, req.src_lang, req.tgt_lang)
        
        # Save to cache
        translation_cache[cache_key] = result
        
        return {
            "translation": result["translation"],
            "latency": result["latency"],
            "character_rate": result["character_rate"],
            "is_cached": False
        }
    except Exception as e:
        print(f"[API SERVER] Translation Error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
