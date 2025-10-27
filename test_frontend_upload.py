#!/usr/bin/env python3
"""
Test frontend-compatible MIDI upload
"""

import requests
import json

def test_frontend_upload():
    print('Testing Frontend MIDI Upload...')
    print('=' * 50)

    try:
        # Create a simple test MIDI data
        test_midi_data = b'MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x80MTrk\x00\x00\x00\x0b\x00\x90\x3c\x40\x00\x40\x80\x3c\x00\x00\xff\x2f\x00'
        
        # Use the same field name as frontend ('file')
        files = {'file': ('test.mid', test_midi_data, 'audio/midi')}
        response = requests.post('http://localhost:8000/api/upload-midi', files=files, timeout=60)
        
        print(f'Upload Status Code: {response.status_code}')
        
        if response.status_code == 200:
            upload = response.json()
            print('SUCCESS! Upload Response:')
            print(f'  Message: {upload.get("message")}')
            print(f'  Filename: {upload.get("filename")}')
            print(f'  File Size: {upload.get("file_size")} bytes')
            print(f'  Duration: {upload.get("duration")} seconds')
            print(f'  Sample Rate: {upload.get("sample_rate")} Hz')
            print(f'  Synthesis Mode: {upload.get("synthesis_mode")}')
            print(f'  Quality: {upload.get("quality")}')
            
            # Test download
            filename = upload.get('filename')
            if filename:
                download_url = f'http://localhost:8000/api/download/{filename}'
                download_response = requests.get(download_url, timeout=30)
                print('\nDownload Test:')
                print(f'  Status: {download_response.status_code}')
                print(f'  Content Length: {len(download_response.content)} bytes')
                
                if download_response.status_code == 200:
                    print('\nSUCCESS: Frontend upload and download working!')
                    return True
                else:
                    print('\nERROR: Download failed')
                    return False
            else:
                print('\nERROR: No filename in response')
                return False
        else:
            print(f'ERROR: Upload failed with status {response.status_code}')
            print(f'Response: {response.text}')
            return False
            
    except Exception as e:
        print(f'ERROR: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_frontend_upload()
    if success:
        print('\nSUCCESS: Frontend upload is working!')
    else:
        print('\nERROR: Frontend upload needs fixing')
