import os
import ffmpeg

input_video_path = "input_video.mp4"
output_audio_path = "audio/extracted_audio.wav"

# Create output directory if it doesn't exist
os.makedirs("audio", exist_ok=True)

print("🎬 Starting audio extraction...")

try:
    (
        ffmpeg
        .input(input_video_path)
        .output(output_audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run()
    )
    print(f"✅ Audio extracted successfully and saved to: {output_audio_path}")
except ffmpeg.Error as e:
    print("❌ FFmpeg error:", e.stderr.decode())
except Exception as e:
    print("⚠️ Unexpected error:", str(e))
