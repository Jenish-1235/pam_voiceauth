#!/usr/bin/env python3
import os
import sys
import json
import wave
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from vosk import Model as VoskModel, KaldiRecognizer
from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.hparams import sampling_rate
from pathlib import Path
from threading import Thread, Event
from queue import Queue

# Constants
CONFIG_PATH = Path.home() / ".voicepam" / "config.json"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk_model")
AUDIO_PATH = "/tmp/voicepam_input.wav"

fail_fast = Event()
result_queue = Queue()

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


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def check_phrase(audio_path):
    if fail_fast.is_set(): return
    print("[phrase] Checking phrase...")
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

    expected = load_config()["phrase"].strip().lower()
    print(f"[phrase] Detected: '{phrase}' | Expected: '{expected}'")

    if phrase != expected:
        fail_fast.set()
        result_queue.put(False)
    else:
        result_queue.put(True)

def check_voice(audio_path):
    if fail_fast.is_set(): return
    print("[voice] Checking voice...")
    encoder = VoiceEncoder()
    audio = preprocess_wav(audio_path)
    embedding = encoder.embed_utterance(audio)

    expected = np.array(load_config()["embedding"])
    similarity = np.dot(embedding, expected) / (np.linalg.norm(embedding) * np.linalg.norm(expected))
    print(f"[voice] Cosine similarity: {similarity:.4f}")

    if similarity < 0.75:
        fail_fast.set()
        result_queue.put(False)
    else:
        result_queue.put(True)

def main():
    try:
        audio_path = record_audio_pyaudio(AUDIO_PATH)

        t1 = Thread(target=check_phrase, args=(audio_path,))
        t2 = Thread(target=check_voice, args=(audio_path,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        if all(results):
            print("[voice_daemon] ✅ Authentication successful")
            sys.exit(0)
        else:
            print("[voice_daemon] ❌ Authentication failed")
            sys.exit(1)

    except Exception as e:
        print(f"[voice_daemon] Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
