#!/usr/bin/env python3
import os
import json
import wave
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from vosk import Model as VoskModel, KaldiRecognizer
from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.hparams import sampling_rate
from pathlib import Path

# Constants
CONFIG_DIR = Path.home() / ".voicepam"
CONFIG_PATH = CONFIG_DIR / "config.json"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk_model")
AUDIO_PATH = "/tmp/voicepam_enroll.wav"

import pyaudio
import wave

def record_audio_pyaudio(path="/tmp/voicepam.wav", duration=3, rate=16000, device_index=None):
    print("🎙️ Recording audio using PyAudio...")

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = 1024

    audio = pyaudio.PyAudio()

    if device_index is None:
        device_index = audio.get_default_input_device_info()['index']

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)

    frames = []
    for _ in range(int(rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"✅ Saved audio to {path}")
    return path


def extract_phrase(audio_path):
    print("🧠 Extracting phrase with Vosk...")
    vosk_model = VoskModel(MODEL_PATH)
    rec = KaldiRecognizer(vosk_model, sampling_rate)

    wf = wave.open(audio_path, "rb")
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result())["text"])
    final = json.loads(rec.FinalResult())["text"]
    results.append(final)
    phrase = " ".join(results).strip().lower()
    print(f"🗣️ Detected phrase: '{phrase}'")
    return phrase

def extract_embedding(audio_path):
    print("🧬 Extracting voice embedding...")
    encoder = VoiceEncoder()
    audio = preprocess_wav(audio_path)
    embedding = encoder.embed_utterance(audio)
    return embedding.tolist()

def save_config(phrase, embedding):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "phrase": phrase,
        "embedding": embedding
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    os.chmod(CONFIG_PATH, 0o600)
    print(f"✅ Config saved to {CONFIG_PATH}")

def main():
    path = record_audio_pyaudio(AUDIO_PATH)
    phrase = extract_phrase(path)
    if not phrase:
        print("❌ Failed to detect phrase. Try again.")
        return
    embedding = extract_embedding(path)
    save_config(phrase, embedding)
    print("🎉 Voice enrollment complete!")

if __name__ == "__main__":
    main()
