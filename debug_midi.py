#!/usr/bin/env python3
"""
Debug MIDI Processing - Deep Audit
"""

import pretty_midi
import librosa
import os

def debug_midi_processing():
    midi_file_path = "MIDI Files/MIDI Files/Cello Arpegio.mid"
    
    print("=== MIDI FILE ANALYSIS ===")
    
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    
    print(f"File: {midi_file_path}")
    print(f"Instruments: {len(midi_data.instruments)}")
    
    # Analyze each instrument
    for i, instrument in enumerate(midi_data.instruments):
        print(f"\nInstrument {i}:")
        print(f"  Program: {instrument.program}")
        print(f"  Notes: {len(instrument.notes)}")
        
        if instrument.notes:
            # Show first 10 notes
            print("  First 10 notes:")
            for j, note in enumerate(instrument.notes[:10]):
                print(f"    {j+1}: pitch={note.pitch}, start={note.start:.2f}s, end={note.end:.2f}s, vel={note.velocity}")
            
            # Calculate total duration
            total_duration = max(note.end for note in instrument.notes)
            print(f"  Total duration: {total_duration:.2f} seconds")
            
            # Analyze pitch range
            pitches = [note.pitch for note in instrument.notes]
            print(f"  Pitch range: {min(pitches)} - {max(pitches)}")
            
            # Analyze velocity range
            velocities = [note.velocity for note in instrument.notes]
            print(f"  Velocity range: {min(velocities)} - {max(velocities)}")
    
    print("\n=== PROCESSING SIMULATION ===")
    
    # Simulate the processing logic
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append({
                'pitch': note.pitch,
                'velocity': note.velocity,
                'start': note.start,
                'end': note.end
            })
    
    print(f"Total notes extracted: {len(notes)}")
    
    # Calculate actual duration
    if notes:
        actual_duration = max(note['end'] for note in notes) + 0.5
        print(f"Calculated duration: {actual_duration:.2f} seconds")
        
        # Show note distribution over time
        print("\nNote distribution:")
        for i in range(0, int(actual_duration) + 1):
            notes_in_second = [n for n in notes if i <= n['start'] < i + 1]
            if notes_in_second:
                print(f"  {i}-{i+1}s: {len(notes_in_second)} notes")
    
    print("\n=== AUDIO GENERATION SIMULATION ===")
    
    # Simulate audio generation
    sr = 48000  # Sample rate
    duration = actual_duration if notes else 2.0
    n_samples = int(duration * sr)
    
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total samples: {n_samples:,}")
    
    # Calculate how many samples would be active
    active_samples = 0
    for note in notes:
        start_sample = int(note['start'] * sr)
        end_sample = int(note['end'] * sr)
        active_samples += max(0, end_sample - start_sample)
    
    print(f"Active samples: {active_samples:,}")
    print(f"Activity ratio: {active_samples/n_samples*100:.1f}%")

if __name__ == '__main__':
    debug_midi_processing()
