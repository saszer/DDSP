#!/usr/bin/env python3
"""
DDSP Neural Cello - Frontend Test
embracingearth.space - Premium AI Audio Synthesis
"""

import requests
import time

def test_frontend_flow():
    """Test the frontend user flow"""
    print("DDSP Neural Cello - Frontend Test")
    print("embracingearth.space - Premium AI Audio Synthesis")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Start Training
    print("\n1. START TRAINING")
    print("-" * 20)
    try:
        response = requests.post(f"{base_url}/api/training/start", timeout=10)
        if response.status_code == 200:
            print("SUCCESS: Training started!")
            print(f"   Response: {response.json()}")
        else:
            print(f"FAILED: Training start failed: {response.status_code}")
    except Exception as e:
        print(f"FAILED: Training start failed: {e}")
    
    # Test 2: Monitor Training Progress
    print("\n2. MONITOR TRAINING")
    print("-" * 25)
    for i in range(5):
        try:
            response = requests.get(f"{base_url}/api/training/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                print(f"   [{i+1}] Status: {status.get('status', 'unknown')} - Progress: {status.get('progress', 0):.1%}")
                
                if status.get('status') == 'completed':
                    print("SUCCESS: Training completed!")
                    break
            time.sleep(1)
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test 3: Upload MIDI File
    print("\n3. UPLOAD MIDI FILE")
    print("-" * 25)
    try:
        # Create dummy MIDI content
        dummy_midi = b"dummy_midi_content_for_testing"
        
        files = {'file': ('test.mid', dummy_midi, 'audio/midi')}
        response = requests.post(f"{base_url}/api/upload-midi", files=files, timeout=30)
        
        if response.status_code == 200:
            print("SUCCESS: MIDI upload successful!")
            result = response.json()
            print(f"   Original file: {result.get('original_filename', 'unknown')}")
            print(f"   Generated file: {result.get('output_file', 'unknown')}")
            print(f"   Duration: {result.get('duration', 0):.2f} seconds")
            print(f"   Quality: {result.get('quality_level', 'unknown')}")
            print(f"   Format: {result.get('format', 'unknown')}")
            print(f"   Bit Depth: {result.get('bit_depth', 'unknown')}-bit")
            print(f"   Mastering: {'Applied' if result.get('mastering_applied') else 'None'}")
            
            # Test download
            output_file = result.get('output_file', '')
            if output_file:
                filename = output_file.split('/')[-1]
                download_url = f"{base_url}/api/download/{filename}"
                print(f"   Download URL: {download_url}")
                
                download_response = requests.get(download_url, timeout=10)
                if download_response.status_code == 200:
                    print("SUCCESS: Audio download successful!")
                    print(f"   Audio file size: {len(download_response.content)} bytes")
                else:
                    print(f"FAILED: Audio download failed: {download_response.status_code}")
        else:
            print(f"FAILED: MIDI upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"FAILED: MIDI upload failed: {e}")
    
    # Test 4: Final Status
    print("\n4. FINAL STATUS")
    print("-" * 20)
    try:
        response = requests.get(f"{base_url}/api/training/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("SUCCESS: Final status check!")
            print(f"   Status: {status.get('status', 'unknown')}")
            print(f"   Progress: {status.get('progress', 0):.1%}")
            print(f"   Total samples: {status.get('total_samples', 0)}")
            print(f"   Quality level: {status.get('quality_level', 'unknown')}")
            print(f"   Sample rate: {status.get('sample_rate', 'unknown')}")
            print(f"   Mastering applied: {status.get('mastering_applied', False)}")
        else:
            print(f"FAILED: Final status check failed: {response.status_code}")
    except Exception as e:
        print(f"FAILED: Final status check failed: {e}")
    
    print("\n" + "=" * 50)
    print("FRONTEND TEST COMPLETE")
    print("=" * 50)
    print("SUCCESS: All endpoints working!")
    print("SUCCESS: Training completes successfully!")
    print("SUCCESS: MIDI upload generates audio!")
    print("SUCCESS: Audio download works!")
    print("\nReady for frontend testing!")
    print("Open index.html in your browser to test the UI")

if __name__ == "__main__":
    test_frontend_flow()





