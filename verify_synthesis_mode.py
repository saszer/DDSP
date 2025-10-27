"""Verify which synthesis mode is being used"""

import requests
import json

# Upload MIDI
with open("MIDI Files/MIDI Files/Cello Arpegio.mid", 'rb') as f:
    files = {'midi_file': ('test.mid', f, 'audio/midi')}
    response = requests.post('http://localhost:8000/api/upload-midi', files=files)

if response.status_code == 200:
    data = response.json()
    print("="*60)
    print("SYNTHESIS MODE VERIFICATION")
    print("="*60)
    print(f"Success: {data.get('success')}")
    print(f"Duration: {data.get('duration'):.2f}s")
    print(f"Output: {data.get('output_file')}")
    print(f"Synthesis Mode: {data.get('synthesis_mode', 'NOT REPORTED')}")
    print("="*60)
    
    if data.get('synthesis_mode') == 'TRAINED_MODEL':
        print("[OK] Using REAL CELLO SAMPLES!")
    elif data.get('synthesis_mode') == 'FALLBACK_HIGH_QUALITY':
        print("[WARNING] Using synthetic fallback (not real samples)")
    else:
        print("[INFO] Synthesis mode unknown")
else:
    print(f"Error: {response.status_code}")
