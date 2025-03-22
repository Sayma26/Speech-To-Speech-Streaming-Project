import ffmpeg

def extract_audio(video_path, output_audio_path):
    (
        ffmpeg
        .input(video_path)
        .output(output_audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run()
    )

extract_audio('sample_video.mp4', 'extracted_audio.wav')
