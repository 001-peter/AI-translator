import sys

# Force system standard streams to use UTF-8 to prevent charmap UnicodeEncodeErrors in Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from translator import TranslationEngine

def main():
    print("Initializing Translation Engine (NLLB-200 with CPU Quantization)...")
    try:
        engine = TranslationEngine(use_quantization=True)
    except Exception as e:
        print(f"Error initializing TranslationEngine: {e}")
        sys.exit(1)
        
    print("\nInitialization successful. Running translation test:")
    
    src_text = "Hello, world! Welcome to local machine translation using PyTorch and Hugging Face."
    src_lang = "eng_Latn"
    tgt_lang = "yor_Latn"
    
    print(f"\nSource text ({src_lang}): {src_text}")
    print("Translating to Yoruba (yor_Latn)...")
    
    try:
        result = engine.translate(src_text, src_lang, tgt_lang)
        print(f"Translation result ({tgt_lang}): {result['translation']}")
        print(f"Latency: {result['latency']:.2f} seconds")
        print(f"Speed: {result['character_rate']:.2f} characters per second")
        
        # Basic validation
        if not result['translation']:
            print("FAILED: Translation is empty.")
            sys.exit(1)
        else:
            print("\nSUCCESS: Translation completed successfully.")
            
    except Exception as e:
        print(f"FAILED during translation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
