# File: generate_csv_video.py

import torch
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import pandas as pd
import os

# --- CONFIGURATION ---
output_dir = r"C:\Users\eazy8\Downloads\backup chatgbt\video\GAIA_Output"
csv_path = r"C:\Users\eazy8\Downloads\backup chatgbt\video\GAIA_Production_Master_Sheet.csv"
os.makedirs(output_dir, exist_ok=True)

# --- 6GB VRAM OPTIMIZED SETTINGS ---
fps = 24
test_mode = False # Always in production mode

# We are reducing the clip length and render quality to fit in 6GB VRAM
num_frames = 48  # 2 seconds (down from 72)
num_inference_steps = 40 # High quality (down from 50)


def make_safe_filename(prompt, shot_number):
    words = prompt.split()[:4]
    safe_name = "_".join(word.lower().strip(".,!?") for word in words)
    return f"shot_{shot_number:03d}_{safe_name}.mp4"

def main():
    try:
        print("Loading diffusion pipeline with stable FP32 precision...")
        pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float32
        )
        pipe.to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_vae_slicing()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading pipeline: {str(e)}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"FATAL: Error loading CSV: {str(e)}")
        return

    print("\n--- STARTING 6GB VRAM OPTIMIZED PRODUCTION RUN ---")
    print(f"Will process {len(df)} shots.")

    for index, row in df.iterrows():
        shot_number = row['Shot#']
        visual_prompt = row['Visual Prompt']
        style_notes = row['Style Notes']

        prompt = f"{visual_prompt}, {style_notes}, cinematic, photorealistic, high detail"
        negative_prompt = "blurry, low quality, cartoon, cgi, text, watermark"

        print(f"\n>>> Generating Shot #{shot_number:03d}: '{visual_prompt}'")
        print(f"    (Using {num_inference_steps} steps for {num_frames} frames)")

        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames
            )
            video_frames = result.frames[0]

            output_filename = make_safe_filename(visual_prompt, shot_number)
            output_path = os.path.join(output_dir, output_filename)
            print(f"    Saving to {output_path}...")
            export_to_video(video_frames, output_path, fps=fps)
            print(f"--- Shot #{shot_number:03d} successfully saved! ---")

        except Exception as e:
            # Catch the Out of Memory error specifically if it happens again
            if "CUDA out of memory" in str(e):
                print(f"!!! VRAM LIMIT HIT on Shot #{shot_number}. The settings are still too high for this shot.")
                print("!!! Skipping to next shot. Consider reducing num_frames further (e.g., to 36).")
            else:
                print(f"!!! ERROR processing Shot #{shot_number}: {str(e)}")
            continue

    print("\n--- All processing complete! ---")

if __name__ == "__main__":
    main()