import os

original_video = "input_video.mp4"  # Your clean, silent video
audio_folder = "audio_outputs"
output_folder = "merged_videos"

languages = [
    "english", "hindi", "marathi", "tamil", "telugu", "bengali",
    "gujarati", "kannada", "malayalam", "punjabi", "urdu", "french"
]

os.makedirs(output_folder, exist_ok=True)

for lang in languages:
    audio_file = os.path.join(audio_folder, f"{lang}.mp3")
    output_file = os.path.join(output_folder, f"{lang}.mp4")

    if os.path.exists(audio_file):
        cmd = (
            f'ffmpeg -y -i "{original_video}" -i "{audio_file}" '
            f'-map 0:v -map 1:a -c:v copy -c:a aac -shortest "{output_file}"'
        )
        print(f"🔁 Merging {lang}.mp3 into video...")
        os.system(cmd)
    else:
        print(f"⚠️ Skipping {lang} — audio file not found!")

print("✅ All videos merged with translated audio!")
