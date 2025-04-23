import pyaudio

p = pyaudio.PyAudio()
info = p.get_device_info_by_index(4)  # use your internal mic index here

print("Device:", info['name'])
print("Default Sample Rate:", info['defaultSampleRate'])
