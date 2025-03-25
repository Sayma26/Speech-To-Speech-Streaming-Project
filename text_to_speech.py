from gtts import gTTS
import os

# Your 12 languages and their corresponding gTTS language codes
gtts_lang_codes = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
    "bengali": "bn",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "punjabi": "pa",
    "urdu": "ur",
    "french": "fr"
}

# Create output directory if not exists
os.makedirs("audio_outputs", exist_ok=True)

# Generate audio for each translated text
for lang, code in gtts_lang_codes.items():
    input_file = f"translated_{lang}.txt"
    output_file = f"audio_outputs/{lang}.mp3"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        print(f"🔊 Converting {lang.capitalize()} text to speech...")
        tts = gTTS(text=text, lang=code)
        tts.save(output_file)
        print(f"✅ Audio saved: {output_file}")

    except Exception as e:
        print(f"❌ Error processing {lang}: {e}")

print("\n🎉 All text-to-speech conversions completed successfully!")
