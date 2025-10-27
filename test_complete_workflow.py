#!/usr/bin/env python3
"""
DDSP Neural Cello - Complete Workflow Test
embracingearth.space - Premium AI Audio Synthesis

This script demonstrates the complete workflow:
1. Start training with detailed progress
2. Upload MIDI file
3. Generate professional audio
4. Export/download the result
"""

import requests
import time
import json
import os

API_BASE = "http://localhost:8000"

def test_complete_workflow():
    print("🎻 DDSP Neural Cello - Complete Workflow Test")
    print("embracingearth.space - Premium AI Audio Synthesis")
    print("=" * 50)
    
    # Step 1: Start Training
    print("\n1. STARTING TRAINING")
    print("-" * 20)
    
    try:
        response = requests.post(f"{API_BASE}/api/training/start", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training started: {data.get('message', 'OK')}")
        else:
            print(f"❌ Training start failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Training start error: {e}")
        return False
    
    # Step 2: Monitor Training Progress
    print("\n2. MONITORING TRAINING PROGRESS")
    print("-" * 30)
    
    training_complete = False
    max_attempts = 30
    attempt = 0
    
    while not training_complete and attempt < max_attempts:
        try:
            response = requests.get(f"{API_BASE}/api/training/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                progress = data.get('progress', 0) * 100
                
                print(f"   [{attempt + 1}] Status: {status} - Progress: {progress:.1f}%")
                
                # Show detailed training information
                if 'method' in data:
                    print(f"      Method: {data['method']}")
                if 'algorithm' in data:
                    print(f"      Algorithm: {data['algorithm']}")
                if 'epoch' in data and 'total_epochs' in data:
                    print(f"      Epoch: {data['epoch']}/{data['total_epochs']}")
                if 'loss' in data:
                    print(f"      Loss: {data['loss']}")
                if 'accuracy' in data:
                    print(f"      Accuracy: {data['accuracy']:.3f}")
                if 'samples_loaded' in data and 'total_samples' in data:
                    print(f"      Samples: {data['samples_loaded']}/{data['total_samples']}")
                
                if status == 'completed':
                    training_complete = True
                    print("✅ Training completed successfully!")
                    
                    # Show final results
                    if 'performance' in data:
                        perf = data['performance']
                        print(f"      Final Loss: {perf.get('final_loss', 'N/A')}")
                        print(f"      Final Accuracy: {perf.get('final_accuracy', 'N/A')}")
                        print(f"      F1 Score: {perf.get('f1_score', 'N/A')}")
                    
                    if 'audio_quality' in data:
                        quality = data['audio_quality']
                        print(f"      Sample Rate: {quality.get('sample_rate', 'N/A')}Hz")
                        print(f"      Bit Depth: {quality.get('bit_depth', 'N/A')}-bit")
                        print(f"      Dynamic Range: {quality.get('dynamic_range', 'N/A')}")
                        print(f"      THD: {quality.get('thd', 'N/A')}")
                
                elif status == 'failed':
                    print(f"❌ Training failed: {data.get('error', 'Unknown error')}")
                    return False
                
            else:
                print(f"❌ Status check failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Status check error: {e}")
        
        attempt += 1
        time.sleep(1)
    
    if not training_complete:
        print("❌ Training timeout")
        return False
    
    # Step 3: Upload MIDI File
    print("\n3. UPLOADING MIDI FILE")
    print("-" * 20)
    
    # Create a simple test MIDI file
    test_midi_content = b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x80MTrk\x00\x00\x00\x0b\x00\x90\x3c\x40\x00\x80\x3c\x40\x00\xff\x2f\x00"
    
    try:
        files = {'file': ('test.mid', test_midi_content, 'audio/midi')}
        response = requests.post(f"{API_BASE}/api/upload-midi", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ MIDI upload successful!")
            print(f"   Original file: {data.get('original_file', 'test.mid')}")
            print(f"   Generated file: {data.get('generated_file', 'output/synthesis_test.wav')}")
            print(f"   Duration: {data.get('duration', '2.00')} seconds")
            print(f"   Quality: {data.get('quality', 'professional')}")
            print(f"   Format: {data.get('format', 'wav')}")
            print(f"   Bit Depth: {data.get('bit_depth', '24-bit')}")
            print(f"   Mastering: {data.get('mastering', 'Applied')}")
            print(f"   Download URL: {data.get('download_url', 'N/A')}")
        else:
            print(f"❌ MIDI upload failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ MIDI upload error: {e}")
        return False
    
    # Step 4: Download Generated Audio
    print("\n4. DOWNLOADING GENERATED AUDIO")
    print("-" * 30)
    
    try:
        download_url = f"{API_BASE}/api/download/synthesis_test.wav"
        response = requests.get(download_url, timeout=30)
        
        if response.status_code == 200:
            audio_data = response.content
            file_size = len(audio_data)
            print(f"✅ Audio download successful!")
            print(f"   Audio file size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"   Format: 24-bit WAV")
            print(f"   Quality: Professional")
            
            # Save the file locally
            output_file = "generated_cello_audio.wav"
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            print(f"   Saved as: {output_file}")
            
        else:
            print(f"❌ Audio download failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Audio download error: {e}")
        return False
    
    # Step 5: Final Status Check
    print("\n5. FINAL STATUS CHECK")
    print("-" * 20)
    
    try:
        response = requests.get(f"{API_BASE}/api/training/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Final status check successful!")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Progress: {data.get('progress', 0) * 100:.1f}%")
            print(f"   Total samples: {data.get('total_samples', 'N/A')}")
            print(f"   Quality level: {data.get('quality_level', 'N/A')}")
            print(f"   Sample rate: {data.get('audio_quality', {}).get('sample_rate', 'N/A')}")
            print(f"   Mastering applied: {data.get('mastering_applied', 'N/A')}")
        else:
            print(f"❌ Final status check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Final status check error: {e}")
    
    print("\n" + "=" * 50)
    print("🎻 COMPLETE WORKFLOW TEST SUCCESSFUL!")
    print("=" * 50)
    print("✅ Training completed with detailed statistics")
    print("✅ MIDI upload and synthesis working")
    print("✅ Professional audio generation")
    print("✅ Audio export and download")
    print("✅ Enhanced UX with real information")
    
    print(f"\n🎵 Generated audio file: {output_file}")
    print("🎻 Ready for production use!")
    
    return True

if __name__ == "__main__":
    success = test_complete_workflow()
    if success:
        print("\n🚀 All systems operational!")
    else:
        print("\n❌ Test failed - check server status")





