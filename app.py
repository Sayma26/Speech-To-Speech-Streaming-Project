from flask import Flask, render_template, request
import os

app = Flask(__name__)

languages = [
    "english", "hindi", "marathi", "tamil", "telugu",
    "bengali", "gujarati", "kannada", "malayalam",
    "punjabi", "urdu", "french"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_video = None
    selected_lang = None

    if request.method == 'POST':
        selected_lang = request.form['language']
        filename = f"{selected_lang}.mp4"
        video_path = os.path.join('static', 'videos', filename)
        if os.path.exists(video_path):
            selected_video = filename

    return render_template('index.html', languages=languages, selected_video=selected_video, selected_lang=selected_lang)

if __name__ == '__main__':
    app.run(debug=True)
