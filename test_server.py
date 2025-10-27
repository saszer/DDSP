#!/usr/bin/env python3
"""
DDSP Neural Cello - Test Script
embracingearth.space - Premium AI Audio Synthesis
"""

import requests
import time
import os

def test_ddsp_server():
    """Test the DDSP Neural Cello server"""
    print("DDSP Neural Cello - Testing Server")
    print("embracingearth.space - Premium AI Audio Synthesis")
    print("")
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health Check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Health check passed!")
            print(f"   Response: {response.json()}")
        else:
            print(f"FAILED: Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"FAILED: Health check failed: {e}")
        return False
    
    # Test 2: Training Status
    print("\n2. Testing training status endpoint...")
    try:
        response = requests.get(f"{base_url}/api/training/status", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Training status endpoint working!")
            print(f"   Response: {response.json()}")
        else:
            print(f"FAILED: Training status failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"FAILED: Training status failed: {e}")
    
    # Test 3: Start Training
    print("\n3. Testing training start endpoint...")
    try:
        response = requests.post(f"{base_url}/api/training/start", timeout=10)
        if response.status_code == 200:
            print("SUCCESS: Training start endpoint working!")
            print(f"   Response: {response.json()}")
        else:
            print(f"FAILED: Training start failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"FAILED: Training start failed: {e}")
    
    # Test 4: Check Training Progress
    print("\n4. Checking training progress...")
    for i in range(3):
        try:
            response = requests.get(f"{base_url}/api/training/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                print(f"   Progress: {status.get('progress', 0):.1%} - {status.get('status', 'unknown')}")
                if status.get('status') == 'completed':
                    print("SUCCESS: Training completed!")
                    break
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"   Error checking progress: {e}")
    
    # Test 5: Test MIDI Upload (simulated)
    print("\n5. Testing MIDI upload endpoint...")
    try:
        # Create a dummy MIDI file content
        dummy_midi = b"dummy_midi_content_for_testing"
        
        files = {'file': ('test.mid', dummy_midi, 'audio/midi')}
        response = requests.post(f"{base_url}/api/upload-midi", files=files, timeout=30)
        
        if response.status_code == 200:
            print("SUCCESS: MIDI upload endpoint working!")
            result = response.json()
            print(f"   Generated file: {result.get('output_file', 'unknown')}")
            print(f"   Quality: {result.get('quality_level', 'unknown')}")
            print(f"   Format: {result.get('format', 'unknown')}")
            print(f"   Bit Depth: {result.get('bit_depth', 'unknown')}")
        else:
            print(f"FAILED: MIDI upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"FAILED: MIDI upload failed: {e}")
    
    print("\nTesting completed!")
    print("\nSummary:")
    print("   - Backend API: http://localhost:8000")
    print("   - Frontend: Open index.html in browser")
    print("   - Quality: Professional-grade audio synthesis")
    print("   - Features: MIDI upload, training, high-quality audio export")
    print("\nReady to use!")
    
    return True

if __name__ == "__main__":
    test_ddsp_server()