import requests
import time

print('DDSP Neural Cello - Complete Workflow Test')
print('=' * 50)

# Start training
print('1. Starting training...')
try:
    r = requests.post('http://localhost:8000/api/training/start', timeout=30)
    print('Training started:', r.status_code == 200)
except:
    print('Training start failed')

# Monitor training
print('2. Monitoring training...')
for i in range(5):
    try:
        r = requests.get('http://localhost:8000/api/training/status', timeout=5)
        data = r.json()
        status = data.get('status', 'unknown')
        progress = data.get('progress', 0) * 100
        print(f'  [{i+1}] Status: {status} - Progress: {progress:.1f}%')
        if data.get('status') == 'completed':
            print('Training completed!')
            break
    except:
        print(f'  [{i+1}] Status check failed')
    time.sleep(1)

# Upload MIDI
print('3. Uploading MIDI...')
try:
    files = {'file': ('test.mid', b'test midi content', 'audio/midi')}
    r = requests.post('http://localhost:8000/api/upload-midi', files=files, timeout=30)
    print('MIDI upload:', r.status_code == 200)
    if r.status_code == 200:
        data = r.json()
        print(f'Generated: {data.get("generated_file", "N/A")}')
        print(f'Duration: {data.get("duration", "N/A")}s')
        print(f'Quality: {data.get("quality", "N/A")}')
except:
    print('MIDI upload failed')

# Download audio
print('4. Downloading audio...')
try:
    r = requests.get('http://localhost:8000/api/download/synthesis_test.wav', timeout=30)
    print('Audio download:', r.status_code == 200)
    if r.status_code == 200:
        print(f'Audio size: {len(r.content):,} bytes')
        with open('test_output.wav', 'wb') as f:
            f.write(r.content)
        print('Saved as: test_output.wav')
except:
    print('Audio download failed')

print('=' * 50)
print('Complete workflow test finished!')





