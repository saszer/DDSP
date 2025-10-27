"""Check if trained model is actually being used"""

import sys
import requests
import io
import pretty_midi

def test_and_trace():
    """Test MIDI upload and capture server logs"""
    
    print("Testing with debug trace...")
    
    # Upload MIDI
    midi_file = "MIDI Files/MIDI Files/Cello Arpegio.mid"
    
    with open(midi_file, 'rb') as f:
        files = {'midi_file': ('test.mid', f, 'audio/midi')}
        response = requests.post('http://localhost:8000/api/upload-midi', files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n[OK] Upload successful")
        print(f"[INFO] Output: {data.get('output_file')}")
        print(f"[INFO] Duration: {data.get('duration')}")
        
        # Try to extract synthesis info from response
        # The server should log which path was used
        print("\n[INFO] Check server terminal for synthesis path logs:")
        print("  Look for: 'Trained synthesis OK' or 'Trained synthesis failed'")
        
        return data
    else:
        print(f"[ERROR] Upload failed: {response.status_code}")
        return None

if __name__ == "__main__":
    test_and_trace()

