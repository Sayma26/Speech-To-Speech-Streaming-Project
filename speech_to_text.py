import whisper
import os

# Load the Whisper model
model = whisper.load_model("base")  # You can also try "small", "medium", or "large"

# Path to the extracted audio
audio_path = "audio/extracted_audio.wav"

# Check if the audio file exists
if not os.path.exists(audio_path):
    print("Audio file not found at:", audio_path)
else:
    # Transcribe audio to text
    print("Transcribing...")
    result = model.transcribe(audio_path)
    
    # Save the text to a file
    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    print("Transcription completed! Check 'transcription.txt'.")
