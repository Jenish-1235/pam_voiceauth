#!/usr/bin/env python3
import os
import json
import wave
import numpy as np
import librosa
import soundfile as sf
from vosk import Model as VoskModel, KaldiRecognizer
from resemblyzer import VoiceEncoder
from pathlib import Path
import pyaudio

# Paths
CONFIG_DIR = Path.home() / ".voicepam"
CONFIG_PATH = CONFIG_DIR / "config.json"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk_model")
AUDIO_PATH = "/tmp/voicepam_enroll.wav"

def record_audio_resampled(out_path=AUDIO_PATH, duration=3, target_sr=16000, device_index=None):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = 1024

    pa = pyaudio.PyAudio()

    if device_index is None:
        device_index = pa.get_default_input_device_info()['index']
    dev_info = pa.get_device_info_by_index(device_index)
    native_sr = int(dev_info['defaultSampleRate'])

    print(f"[mic] Recording from device {device_index} ({dev_info['name']}) at {native_sr} Hz")

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

    # Convert to float32
    audio_bytes = b''.join(frames)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

    # Resample to 16kHz
    audio_resampled = librosa.resample(audio_np, orig_sr=native_sr, target_sr=target_sr)

    # Save as 16kHz WAV for Vosk
    sf.write(out_path, audio_resampled, target_sr)
    print(f"[mic] Saved resampled audio to {out_path}")

    return audio_resampled

def extract_phrase(wav_path):
    print("🧠 Extracting phrase with Vosk...")
    model = VoskModel(MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)

    wf = wave.open(wav_path, "rb")
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

def extract_embedding(audio_array):
    print("🧬 Extracting voice embedding...")
    encoder = VoiceEncoder()
    return encoder.embed_utterance(audio_array).tolist()

def save_config(phrase, embedding):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "phrase": phrase,
            "embedding": embedding
        }, f, indent=2)
    os.chmod(CONFIG_PATH, 0o600)
    print(f"✅ Voice config saved to {CONFIG_PATH}")

def main():
    audio_array = record_audio_resampled()
    phrase = extract_phrase(AUDIO_PATH)

    if not phrase:
        print("❌ Could not detect a valid phrase. Please try again.")
        return

    embedding = extract_embedding(audio_array)
    save_config(phrase, embedding)
    print("🎉 Voice enrollment complete! You can now log in with your voice.")

if __name__ == "__main__":
    main()
