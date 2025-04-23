#!/usr/bin/env python3
import os
import sys
import json
import wave
import librosa
import numpy as np
import soundfile as sf
import pyaudio
from vosk import Model as VoskModel, KaldiRecognizer
from resemblyzer import VoiceEncoder
from pathlib import Path
from threading import Thread, Event
from queue import Queue
from Levenshtein import ratio

# Paths and constants
CONFIG_PATH = Path.home() / ".voicepam" / "config.json"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk_model")
AUDIO_PATH = "/tmp/voicepam_input.wav"
FAIL_FAST = Event()
RESULT_QUEUE = Queue()

def record_audio_resampled(out_path=AUDIO_PATH, duration=3, target_sr=16000, device_index=None):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = 1024

    pa = pyaudio.PyAudio()
    if device_index is None:
        device_index = pa.get_default_input_device_info()['index']
    dev_info = pa.get_device_info_by_index(device_index)
    native_sr = int(dev_info['defaultSampleRate'])

    print(f"[mic] Using device {device_index} ({dev_info['name']}) at {native_sr} Hz")

    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=native_sr,
                     input=True,
                     input_device_index=device_index,
                     frames_per_buffer=CHUNK)

    frames = []
    for _ in range(int(native_sr / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    audio_bytes = b''.join(frames)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    audio_resampled = librosa.resample(audio_np, orig_sr=native_sr, target_sr=target_sr)

    sf.write(out_path, audio_resampled, target_sr)
    print(f"[mic] Saved resampled audio to {out_path}")

    return audio_resampled

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def check_phrase(audio_path):
    if FAIL_FAST.is_set(): return

    print("[phrase] Checking phrase...")
    model = VoskModel(MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)

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
    similarity = ratio(phrase, expected)
    print(f"[phrase] Detected: '{phrase}' | Expected: '{expected}' | Similarity: {similarity:.3f}")

    if similarity < 0.75:
        FAIL_FAST.set()
        RESULT_QUEUE.put(False)
    else:
        RESULT_QUEUE.put(True)

def check_voice(audio_array):
    if FAIL_FAST.is_set(): return

    print("[voice] Checking voice...")
    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(audio_array)

    expected = np.array(load_config()["embedding"])
    similarity = np.dot(embedding, expected) / (np.linalg.norm(embedding) * np.linalg.norm(expected))
    print(f"[voice] Cosine similarity: {similarity:.4f}")

    if similarity < 0.75:
        FAIL_FAST.set()
        RESULT_QUEUE.put(False)
    else:
        RESULT_QUEUE.put(True)

def main():
    try:
        audio_array = record_audio_resampled()

        t1 = Thread(target=check_phrase, args=(AUDIO_PATH,))
        t2 = Thread(target=check_voice, args=(audio_array,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        results = []
        while not RESULT_QUEUE.empty():
            results.append(RESULT_QUEUE.get())

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
