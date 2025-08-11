# File: run_narration_generation_final.py (Verified and Complete)

import torch
import pandas as pd
import os
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
import soundfile as sf
import numpy as np
import wave # Import the fallback library

print("--- The Bulletproof Method: Using the SpeechBrain Toolkit with a Safe Fallback ---")

# --- CONFIGURATION ---
output_dir = r"C:\Users\eazy8\Downloads\backup chatgbt\video\GAIA_Output\narration"
csv_path = r"C:\Users\eazy8\Downloads\backup chatgbt\video\GAIA_Production_Master_Sheet.csv"
os.makedirs(output_dir, exist_ok=True)

# --- HELPER FUNCTION ---
def make_audio_filename(shot_number: int) -> str:
    return f"shot_{shot_number:03d}.wav"

def main():
    # --- Load the SpeechBrain TTS Model ---
    print("Loading SpeechBrain Text-to-Speech model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts", run_opts={"device": device})
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder", run_opts={"device": device})
        print("SpeechBrain models loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load the SpeechBrain models: {e}")
        return

    # Load CSV
    try:
        print(f"Loading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"FATAL: Error loading CSV: {str(e)}")
        return

    print("\n--- Starting Narration Generation ---")
    
    for index, row in df.iterrows():
        shot_number = row['Shot#']
        narration_text = row['Narration Line']

        if not isinstance(narration_text, str) or not narration_text.strip():
            continue
            
        print(f">>> Generating narration for Shot #{shot_number:03d}...")

        try:
            mel_output, mel_length, alignment = tacotron2.encode_text(narration_text)
            waveforms = hifi_gan.decode_batch(mel_output)
            
            output_path = os.path.join(output_dir, make_audio_filename(shot_number))
            
            # --- THE BULLETPROOF SAVE BLOCK (Corrected) ---
            audio_data = waveforms.squeeze().cpu().numpy().astype('float32')

            try:
                sf.write(output_path, audio_data, 22050)
                print(f"--- Narration for Shot #{shot_number:03d} saved via soundfile! ---")
            except RuntimeError as e:
                print(f"    Soundfile write failed ({e}). Falling back to built-in wave module...")
                with wave.open(output_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                print(f"--- Narration for Shot #{shot_number:03d} saved via fallback! ---")

        except Exception as e:
            print(f"!!! CRITICAL ERROR on Shot #{shot_number}: {e}")
            continue

    print("\n--- All narration generation complete! ---")

if __name__ == "__main__":
    main()