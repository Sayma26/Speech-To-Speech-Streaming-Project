from flask import Flask, request, render_template, send_file
import os
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

FINAL_VIDEO = os.path.join(OUTPUT_FOLDER, 'final_video.mp4')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    # Get user inputs
    video_url = request.form['video_url']
    target_language = request.form['language']

    # Define file paths
    video_path = os.path.join(UPLOAD_FOLDER, 'video.mp4')
    audio_path = os.path.join(UPLOAD_FOLDER, 'audio.wav')
    translated_audio_path = os.path.join(OUTPUT_FOLDER, 'translated_audio.mp3')

    # Step 1: Download video
    try:
        subprocess.run(['yt-dlp', '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]', '-o', video_path, video_url], check=True)
    except subprocess.CalledProcessError as e:
        return f"Error downloading video: {str(e)}"

    # Step 2: Extract audio
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
    except Exception as e:
        return f"Error extracting audio: {str(e)}"

    # Step 3: Convert audio to text (Speech Recognition)
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        original_text = recognizer.recognize_google(audio_data)
    except Exception as e:
        return f"Error in speech recognition: {str(e)}"

    # Step 4: Translate text
    translator = Translator()
    try:
        translated_text = translator.translate(original_text, dest=target_language).text
    except Exception as e:
        return f"Error in translation: {str(e)}"

    # Step 5: Convert translated text to speech (gTTS)
    try:
        tts = gTTS(text=translated_text, lang=target_language)
        tts.save(translated_audio_path)
    except Exception as e:
        return f"Error generating speech: {str(e)}"

    # Step 6: Merge translated speech with video
    try:
        new_audio = AudioFileClip(translated_audio_path)
        final_clip = video_clip.set_audio(new_audio)
        final_clip.write_videofile(FINAL_VIDEO, fps=video_clip.fps if video_clip.fps else 30)
    except Exception as e:
        return f"Error merging audio and video: {str(e)}"

    return render_template(
    'result.html',
    youtube_id=video_url.split('v=')[-1],
    original_text=original_text,
    translated_text=translated_text,
    target_language=target_language
)



@app.route('/download')
def download_video():
    if os.path.exists(FINAL_VIDEO):
        return send_file(FINAL_VIDEO, as_attachment=True)
    else:
        return "Error: Final video file not found.", 500

if __name__ == '__main__':
    app.run(debug=True)
