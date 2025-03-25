from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Your selected 12 target languages
languages = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "marathi": "mar_Deva",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "bengali": "ben_Beng",
    "gujarati": "guj_Gujr",
    "kannada": "kan_Knda",
    "malayalam": "mal_Mlym",
    "punjabi": "pan_Guru",
    "urdu": "urd_Arab",
    "french": "fra_Latn"
}

# Read the transcribed text
with open("transcription.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Translate and save
for lang, lang_code in languages.items():
    print(f"Translating to {lang.capitalize()}...")

    tokenizer.src_lang = "eng_Latn"  # source language
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang_code),
        max_length=512
    )
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    with open(f"translated_{lang}.txt", "w", encoding="utf-8") as f:
        f.write(translated)

    print(f"{lang.capitalize()} translation saved to translated_{lang}.txt ✅")

print("🎉 All translations completed!")
