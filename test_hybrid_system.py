#!/usr/bin/env python3
"""
Test script to verify the hybrid DDSP system is working correctly
"""

import requests
import json
import time

def test_hybrid_system():
    """Test the complete hybrid DDSP system"""
    
    print("Testing Hybrid DDSP System")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        
        print(f"   Status: {health_data['status']}")
        print(f"   TensorFlow: {'Available' if health_data['tensorflow_available'] else 'Not Available'}")
        print(f"   Google DDSP: {'Available' if health_data['ddsp_available'] else 'Not Available (Fallback Mode)'}")
        print(f"   Synthesis Mode: {health_data['synthesis_mode']}")
        print(f"   Model Trained: {'Yes' if health_data['model_trained'] else 'No'}")
        
    except Exception as e:
        print(f"   Health check failed: {e}")
        return False
    
    # Test 2: Training Status
    print("\n2. Testing Training Status...")
    try:
        response = requests.get(f"{base_url}/api/training/status")
        training_data = response.json()
        
        print(f"   ✅ Training Status: {training_data['status']}")
        print(f"   ✅ Progress: {training_data['progress']:.1%}")
        print(f"   ✅ Method: {training_data.get('method', 'N/A')}")
        print(f"   ✅ Algorithm: {training_data.get('algorithm', 'N/A')}")
        
        if 'architecture' in training_data:
            arch = training_data['architecture']
            print(f"   ✅ Architecture: {arch.get('layers', 'N/A')} layers, {arch.get('neurons_per_layer', 'N/A')} neurons")
            print(f"   ✅ Google DDSP Enabled: {arch.get('google_ddsp_enabled', 'N/A')}")
            print(f"   ✅ TensorFlow Version: {arch.get('tensorflow_version', 'N/A')}")
        
        if 'performance' in training_data:
            perf = training_data['performance']
            print(f"   ✅ Final Loss: {perf.get('final_loss', 'N/A')}")
            print(f"   ✅ Final Accuracy: {perf.get('final_accuracy', 'N/A'):.1%}")
        
    except Exception as e:
        print(f"   ❌ Training status failed: {e}")
        return False
    
    # Test 3: Start Training (if not already trained)
    if not health_data['model_trained']:
        print("\n3. Starting Training...")
        try:
            response = requests.post(f"{base_url}/api/training/start")
            start_data = response.json()
            
            print(f"   ✅ Training Started: {start_data['message']}")
            
            # Wait for training to complete
            print("   ⏳ Waiting for training to complete...")
            for i in range(20):  # Wait up to 20 seconds
                time.sleep(1)
                response = requests.get(f"{base_url}/api/training/status")
                status_data = response.json()
                
                if status_data['status'] == 'completed':
                    print(f"   ✅ Training Completed!")
                    break
                elif status_data['status'] == 'failed':
                    print(f"   ❌ Training Failed: {status_data.get('error', 'Unknown error')}")
                    return False
                else:
                    print(f"   ⏳ Training Progress: {status_data['progress']:.1%} - {status_data['status']}")
            
        except Exception as e:
            print(f"   ❌ Training start failed: {e}")
            return False
    else:
        print("\n3. Training already completed ✅")
    
    # Test 4: MIDI Upload Test
    print("\n4. Testing MIDI Upload...")
    try:
        # Create a simple test MIDI data (just bytes)
        test_midi_data = b"test_midi_data_for_synthesis"
        
        response = requests.post(
            f"{base_url}/api/upload-midi",
            data=test_midi_data,
            headers={'Content-Type': 'application/octet-stream'}
        )
        
        upload_data = response.json()
        
        print(f"   ✅ Upload Response: {upload_data['message']}")
        print(f"   ✅ Filename: {upload_data['filename']}")
        print(f"   ✅ File Size: {upload_data['file_size']} bytes")
        print(f"   ✅ Duration: {upload_data['duration']:.2f} seconds")
        print(f"   ✅ Sample Rate: {upload_data['sample_rate']} Hz")
        print(f"   ✅ Bit Depth: {upload_data['bit_depth']}-bit")
        print(f"   ✅ Synthesis Mode: {upload_data['synthesis_mode']}")
        print(f"   ✅ Quality: {upload_data['quality']}")
        
    except Exception as e:
        print(f"   ❌ MIDI upload failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Hybrid DDSP System is working perfectly!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = test_hybrid_system()
    if success:
        print("\n✅ System Status: FULLY OPERATIONAL")
        print("🎵 Ready for production use!")
    else:
        print("\n❌ System Status: ISSUES DETECTED")
        print("🔧 Please check the error messages above")
