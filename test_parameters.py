"""Test parameter passing - embracingearth.space"""

import requests
import os

# Test with different parameters
print("Testing parameter passing...")
print("="*60)

# Upload with release=50 and tone=bright
midi_file = "MIDI Files/MIDI Files/Cello Arpegio.mid"

formData = requests.models.RequestEncodingMixin()
files = {'file': ('test.mid', open(midi_file, 'rb'), 'audio/midi')}
data = {
    'release_percent': '50',
    'tone': 'bright'
}

response = requests.post('http://localhost:8000/api/upload-midi', files=files, data=data, timeout=30)

if response.status_code == 200:
    result = response.json()
    print("[OK] Upload successful!")
    print(f"[INFO] Duration: {result.get('duration'):.2f}s")
    print(f"[INFO] Synthesis mode: {result.get('synthesis_mode')}")
    
    # Download to verify
    output_file = result.get('output_file')
    if output_file:
        download_url = f"http://localhost:8000/api/download/{output_file}"
        audio_response = requests.get(download_url)
        if audio_response.status_code == 200:
            print(f"[OK] Audio downloaded: {len(audio_response.content)} bytes")
        else:
            print(f"[ERROR] Download failed: {audio_response.status_code}")
else:
    print(f"[ERROR] Upload failed: {response.status_code}")
    print(response.text)

print("="*60)

