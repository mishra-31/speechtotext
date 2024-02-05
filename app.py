from flask import Flask, request, jsonify
from subprocess import CalledProcessError, run
import numpy as np
import torch
import whisper

app = Flask(__name__)

SAMPLE_RATE = 16000
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE

def load_audio(file: str, sr: int = SAMPLE_RATE):
    print('Loading Audio.........')

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]

    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0



def transcribe(audio):
    model = whisper.load_model('small')
    print('Model Loaded......')

    print('Transcribing.....')
    result = model.transcribe(audio,fp16=torch.cuda.is_available())
    transcription = result['text']

    return transcription


@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    try:
        file = request.json['audio']
        audio = load_audio(file)
        transcription = transcribe(audio)
        return jsonify({'transcription': transcription})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
