"""
Multilingual Book Dataset Normalization Pipeline with Auto-Retries and Chunked Translation
-------------------------------------------------------------------------------------------

- Detects language (langid)
- Romanizes Japanese, Korean, Chinese for title/author fields
- Translates non-English text to English
- Retries failed translations automatically (3 attempts)
- Handles long texts by splitting into chunks
- Uses in-memory translation cache
- Saves final CSV in a single step
"""

import time
import pandas as pd
import langid
from deep_translator import GoogleTranslator
from pypinyin import lazy_pinyin
from korean_romanizer.romanizer import Romanizer
import pykakasi
from tqdm import tqdm

# -----------------------
# CONFIGURATION
# -----------------------
INPUT_FILE = "data/final/final_data.csv"
OUTPUT_FILE = "data/final/books_translated.csv"

COLUMNS_TO_PROCESS = ["title", "author", "description", "publisher"]
MAX_RETRIES = 3          # number of retry attempts for failed translation
RETRY_DELAY = 2          # seconds to wait between retries
CHUNK_SIZE = 4500        # maximum characters per translation chunk

# fix some unusual language codes
LANG_FIX = {
    "nn": "no",   "nb": "no",   "oc": "fr",   "ca": "es",
    "gd": "ga",   "fy": "nl",   "gl": "es",   "ast": "es",
}

# -----------------------
# UTILITY FUNCTIONS
# -----------------------
def fix_lang_code(lang: str) -> str:
    return LANG_FIX.get(lang, lang)

def detect_language_safe(text: str) -> str:
    """Detect language safely, defaulting to 'unknown'."""
    if not text or pd.isna(text):
        return "unknown"
    # if all ASCII letters, assume English
    if all(ord(c) < 128 for c in text) and any(c.isalpha() for c in text):
        return "en"
    try:
        lang, _ = langid.classify(text)
        return fix_lang_code(lang)
    except:
        return "unknown"

# -----------------------
# ROMANIZATION FUNCTIONS
# -----------------------
_kks_converter = None

def romanize_japanese(text: str) -> str:
    """Convert Japanese text to Hepburn romanization."""
    global _kks_converter
    if _kks_converter is None:
        kks = pykakasi.kakasi()
        kks.setMode("H", "a")
        kks.setMode("K", "a")
        kks.setMode("J", "a")
        _kks_converter = kks.getConverter()
    return _kks_converter.do(text)

def romanize_korean(text: str) -> str:
    """Convert Korean text to romanized form."""
    return Romanizer(text).romanize()

def chinese_to_pinyin(text: str) -> str:
    """Convert Chinese text to Pinyin."""
    return " ".join(lazy_pinyin(text))

# -----------------------
# TRANSLATION WITH RETRIES
# -----------------------
translation_cache = {}

def translate_text_with_retries(text: str, target: str = "en") -> str:
    """Translate text to English with automatic retries if failed."""
    if not text or text.strip() == "":
        return text
    if text in translation_cache:
        return translation_cache[text]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            translated = GoogleTranslator(source="auto", target=target).translate(text)
            translation_cache[text] = translated
            return translated
        except Exception as e:
            print(f"[WARN] Translation failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print("[WARN] Returning original text after max retries")
                translation_cache[text] = text
                return text

# -----------------------
# HANDLE LONG TEXTS
# -----------------------
def translate_large_text(text: str, target: str = "en") -> str:
    """Split long text into chunks for translation to avoid API limits."""
    if not text or text.strip() == "":
        return text
    if text in translation_cache:
        return translation_cache[text]

    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    translated_chunks = []

    for chunk in chunks:
        translated_chunks.append(translate_text_with_retries(chunk, target))

    result = " ".join(translated_chunks)
    translation_cache[text] = result
    return result

# -----------------------
# PROCESS EACH CELL
# -----------------------
def process_cell(text, col):
    """Detect language, romanize if needed, and translate non-English text."""
    if not text or str(text).strip() == "":
        return text
    text = str(text)
    lang = detect_language_safe(text)

    # Romanize only for title and author columns
    if col in ["title", "author"]:
        if lang == "ja": return romanize_japanese(text)
        if lang == "ko": return romanize_korean(text)
        if lang.startswith("zh"): return chinese_to_pinyin(text)

    # Translate if language is not English
    if lang not in ["en", "unknown"]:
        return translate_large_text(text, target="en")
    return text

# -----------------------
# MAIN PIPELINE
# -----------------------
if __name__ == "__main__":
    print("🚀 Starting multilingual normalization pipeline with auto-retries and chunked translation...")
    df = pd.read_csv(INPUT_FILE, dtype=str, low_memory=False)

    for col in COLUMNS_TO_PROCESS:
        if col not in df.columns:
            df[col] = ""
        print(f"⚙️ Processing column: {col}")
        df[col] = [process_cell(v, col) for v in tqdm(df[col], desc=col)]

    print(f"💾 Saving translated CSV to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print("✅ Translation and romanization completed.")
