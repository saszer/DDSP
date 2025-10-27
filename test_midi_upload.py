#!/usr/bin/env python3
"""
Test MIDI Upload and Audio Generation
embracingearth.space - DDSP Neural Cello Test
"""

import requests
import os
import json

def test_midi_upload():
    """Test MIDI upload and audio generation"""
    print("Testing MIDI Upload and Audio Generation")
    print("embracingearth.space - DDSP Neural Cello")
    print("=" * 50)
    
    # Test health endpoint first
    print("1. Testing Health Endpoint...")
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        health = response.json()
        print(f"   [OK] Backend Status: {health['status']}")
        print(f"   [OK] Service: {health['service']}")
        print(f"   [OK] Version: {health['version']}")
    except Exception as e:
        print(f"   [ERROR] Health check failed: {e}")
        return False
    
    # Test MIDI upload
    print("\n2. Testing MIDI Upload...")
    midi_file_path = "MIDI Files/MIDI Files/Cello Arpegio.mid"
    
    if not os.path.exists(midi_file_path):
        print(f"   [ERROR] MIDI file not found: {midi_file_path}")
        return False
    
    try:
        with open(midi_file_path, 'rb') as f:
            files = {'midi_file': ('Cello Arpegio.mid', f, 'audio/midi')}
            response = requests.post('http://localhost:8000/api/upload-midi', files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   [OK] Upload successful!")
            print(f"   [OK] Original filename: {result.get('original_filename', 'N/A')}")
            print(f"   [OK] Output file: {result.get('output_file', 'N/A')}")
            print(f"   [OK] Duration: {result.get('duration', 0):.2f} seconds")
            print(f"   [OK] Quality level: {result.get('quality_level', 'N/A')}")
            print(f"   [OK] Format: {result.get('format', 'N/A')}")
            print(f"   [OK] Bit depth: {result.get('bit_depth', 'N/A')}-bit")
            print(f"   [OK] Mastering applied: {result.get('mastering_applied', 'N/A')}")
            
            # Test download
            print("\n3. Testing Audio Download...")
            output_filename = result.get('output_file', '').split('/')[-1]
            if output_filename:
                download_url = f'http://localhost:8000/api/download/{output_filename}'
                download_response = requests.get(download_url, timeout=30)
                
                if download_response.status_code == 200:
                    print(f"   [OK] Download successful!")
                    print(f"   [OK] File size: {len(download_response.content)} bytes")
                    
                    # Save the file locally for verification
                    with open(f"test_output_{output_filename}", 'wb') as f:
                        f.write(download_response.content)
                    print(f"   [OK] Audio file saved as: test_output_{output_filename}")
                    
                    return True
                else:
                    print(f"   [ERROR] Download failed: {download_response.status_code}")
            else:
                print("   [ERROR] No output filename in response")
        else:
            print(f"   [ERROR] Upload failed: {response.status_code}")
            print(f"   [ERROR] Response: {response.text}")
            
    except Exception as e:
        print(f"   [ERROR] Upload test failed: {e}")
    
    return False

def test_training_status():
    """Test training status endpoint"""
    print("\n4. Testing Training Status...")
    try:
        response = requests.get('http://localhost:8000/api/training/status', timeout=10)
        status = response.json()
        print(f"   [OK] Training Status: {status.get('status', 'N/A')}")
        print(f"   [OK] Progress: {status.get('progress', 0):.1%}")
        return True
    except Exception as e:
        print(f"   [ERROR] Training status failed: {e}")
        return False

def main():
    """Run the complete test"""
    print("Starting DDSP Neural Cello Full Test")
    print("embracingearth.space - Premium AI Audio Synthesis")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_midi_upload():
        tests_passed += 1
    
    if test_training_status():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("DDSP NEURAL CELLO TEST COMPLETE!")
    print(f"[OK] Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("SYSTEM STATUS: FULLY OPERATIONAL!")
        print("MIDI to Audio synthesis is working perfectly!")
        print("embracingearth.space - Premium AI Audio Technology")
    else:
        print("SYSTEM STATUS: SOME ISSUES DETECTED")
        print("Please check the error messages above")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
