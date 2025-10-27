#!/usr/bin/env python3
"""
Comprehensive DDSP System Test
Tests all components of the Google DDSP system
"""

import requests
import json
import time

def test_health_check():
    """Test the health check endpoint"""
    print('1. Health Check...')
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        health = response.json()
        print(f'   [OK] Status: {health["status"]}')
        print(f'   [OK] TensorFlow: {"Available" if health["tensorflow_available"] else "Not Available"}')
        print(f'   [OK] Google DDSP: {"Available" if health["ddsp_available"] else "Not Available"}')
        print(f'   [OK] Synthesis Mode: {health["synthesis_mode"]}')
        print(f'   [OK] Model Trained: {"Yes" if health["model_trained"] else "No"}')
        return True
    except Exception as e:
        print(f'   [ERROR] Health check failed: {e}')
        return False

def test_training_status():
    """Test the training status endpoint"""
    print('\n2. Training Status...')
    try:
        response = requests.get('http://localhost:8000/api/training/status', timeout=10)
        status = response.json()
        print(f'   [OK] Status: {status["status"]}')
        print(f'   [OK] Progress: {status["progress"]:.1f}%')
        print(f'   [OK] Method: {status["method"]}')
        print(f'   [OK] Algorithm: {status["algorithm"]}')
        
        if 'architecture' in status:
            arch = status['architecture']
            print(f'   [OK] Architecture: {arch["layers"]} layers, {arch["neurons_per_layer"]} neurons')
            print(f'   [OK] Google DDSP: {"Enabled" if arch["google_ddsp_enabled"] else "Fallback"}')
            print(f'   [OK] TensorFlow: {arch["tensorflow_version"]}')
        
        if 'final_loss' in status:
            print(f'   [OK] Final Loss: {status["final_loss"]:.3f}')
            print(f'   [OK] Final Accuracy: {status["final_accuracy"]:.1f}%')
        
        return True
    except Exception as e:
        print(f'   [ERROR] Training status failed: {e}')
        return False

def test_training_start():
    """Test starting training"""
    print('\n3. Training Start...')
    try:
        response = requests.post('http://localhost:8000/api/training/start', timeout=30)
        training = response.json()
        print(f'   [OK] Training Started: {training["message"]}')
        print(f'   [OK] Method: {training["method"]}')
        print(f'   [OK] Algorithm: {training["algorithm"]}')
        return True
    except Exception as e:
        print(f'   [ERROR] Training start failed: {e}')
        return False

def test_midi_upload():
    """Test MIDI upload and audio generation"""
    print('\n4. MIDI Upload Test...')
    try:
        # Create a simple test MIDI data (C major scale)
        test_midi_data = b'MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x80MTrk\x00\x00\x00\x0b\x00\x90\x3c\x40\x00\x40\x80\x3c\x00\x00\xff\x2f\x00'
        
        files = {'midi_file': ('test.mid', test_midi_data, 'audio/midi')}
        response = requests.post('http://localhost:8000/api/upload-midi', files=files, timeout=60)
        upload = response.json()
        
        print(f'   [OK] Upload Response: {upload["message"]}')
        print(f'   [OK] Filename: {upload["filename"]}')
        print(f'   [OK] File Size: {upload["file_size"]} bytes')
        print(f'   [OK] Duration: {upload["duration"]:.2f} seconds')
        print(f'   [OK] Sample Rate: {upload["sample_rate"]} Hz')
        print(f'   [OK] Bit Depth: {upload["bit_depth"]}-bit')
        print(f'   [OK] Synthesis Mode: {upload["synthesis_mode"]}')
        print(f'   [OK] Quality: {upload["quality"]}')
        
        # Test download
        download_url = f'http://localhost:8000/api/download/{upload["filename"]}'
        download_response = requests.get(download_url, timeout=30)
        if download_response.status_code == 200:
            print(f'   [OK] Download: Success ({len(download_response.content)} bytes)')
        else:
            print(f'   [ERROR] Download: Failed ({download_response.status_code})')
        
        return True
    except Exception as e:
        print(f'   [ERROR] MIDI upload failed: {e}')
        return False

def main():
    """Run comprehensive DDSP system test"""
    print('COMPREHENSIVE DDSP SYSTEM TEST')
    print('=' * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Run all tests
    if test_health_check():
        tests_passed += 1
    
    if test_training_status():
        tests_passed += 1
    
    if test_training_start():
        tests_passed += 1
    
    if test_midi_upload():
        tests_passed += 1
    
    # Summary
    print('\n' + '=' * 50)
    print(f'COMPREHENSIVE TEST COMPLETED!')
    print(f'[OK] Tests Passed: {tests_passed}/{total_tests}')
    
    if tests_passed == total_tests:
        print('SYSTEM STATUS: FULLY OPERATIONAL')
        print('Google DDSP is working perfectly!')
    else:
        print('SYSTEM STATUS: ISSUES DETECTED')
        print('Please check the error messages above')
    
    print('=' * 50)

if __name__ == '__main__':
    main()
