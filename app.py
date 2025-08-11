# app.py 

import os
import io
import zipfile
import tempfile
from typing import Tuple, List

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import gradio as gr
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# --- CONFIGURATION ---
DEFAULT_FPS = 24
DEFAULT_TEST_NUM_FRAMES = 24  # 1 second for testing
DEFAULT_FULL_NUM_FRAMES = 72  # 3 seconds for full run
DEFAULT_TEST_STEPS = 35
DEFAULT_FULL_STEPS = 50

# --- UTILITIES ---
def make_safe_filename(prompt: str, shot_number: int) -> str:
    words = prompt.split()[:5]
    safe = "_".join(w.lower().strip(".,!?()[]{}'\"/") for w in words)
    return f"shot_{shot_number:03d}_{safe}.mp4"

def csv_to_df(csv_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(csv_bytes))
    except Exception:
        return pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")))

# --- THE CRITICAL FIX: A STABLE MODEL LOADER ---
def load_pipeline() -> Tuple[DiffusionPipeline, str]:
    """
    Loads the pipeline using our proven stable configuration (FP32 on CUDA).
    This function replaces the complex, unstable logic.
    """
    model_id = "damo-vilab/text-to-video-ms-1.7b"
    status_msgs = []

    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: This Space requires a GPU, but CUDA is not available.")

    try:
        status_msgs.append(f"Loading {model_id} with stable torch.float32 precision...")
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe.to("cuda")
        status_msgs.append("Model loaded successfully onto GPU.")

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        status_msgs.append("Using DPMSolverMultistepScheduler.")

        pipe.enable_vae_slicing()
        status_msgs.append("Enabled VAE slicing for VRAM efficiency.")
        
        return pipe, "\n".join(status_msgs)
    except Exception as e:
        status_msgs.append(f"FATAL ERROR during model loading: {e}")
        raise RuntimeError("Could not load the pipeline:\n" + "\n".join(status_msgs))

# --- GENERATION CORE ---
def generate_shot(pipe: DiffusionPipeline, prompt: str, negative_prompt: str, num_frames: int, num_inference_steps: int, seed: int) -> List[np.ndarray]:
    generator = torch.manual_seed(seed) if seed is not None else None
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator
    )
    return result.frames[0]

# --- GRADIO INTERFACE LOGIC ---
def run_generation(csv_file, test_mode: bool, test_shot_number: int, fps: int, seed: int):
    # Use a temporary directory that Gradio can access
    with tempfile.TemporaryDirectory() as out_dir:
        logs = []
        def log(s):
            print(s)
            logs.append(s)
            yield "\n".join(logs), "In Progress...", None

        # Load CSV from Gradio file object
        if csv_file is None:
            yield "Please upload a CSV file.", "Error", None
            return
        with open(csv_file.name, "rb") as f:
            df = csv_to_df(f.read())

        # Validate columns
        required_cols = ['Shot#', 'Narration Line', 'Visual Prompt', 'Style Notes']
        for col in required_cols:
            if col not in df.columns:
                df[col] = "" # Add empty column if missing

        # Determine shots to process
        shots_df = df[df['Shot#'] == int(test_shot_number)] if test_mode else df
        if shots_df.empty:
            yield "No shots found for the given selection.", "Error", None
            return

        # Load model
        try:
            pipe, status = load_pipeline()
            yield from log(status)
        except Exception as e:
            yield from log(str(e))
            yield "\n".join(logs), "ERROR: Model Failed to Load", None
            return
        
        generated_files = []

        # Iterate through shots
        for _, row in tqdm(shots_df.iterrows(), total=len(shots_df), desc="Generating Shots"):
            shot_number = int(row['Shot#'])
            visual_prompt = str(row.get('Visual Prompt', '')).strip()
            style_notes = str(row.get('Style Notes', '')).strip()
            prompt = f"{visual_prompt}, {style_notes}, cinematic, photorealistic, high detail"
            negative_prompt = "blurry, low quality, cartoon, cgi, text, watermark"

            num_frames = DEFAULT_TEST_NUM_FRAMES if test_mode else DEFAULT_FULL_NUM_FRAMES
            num_steps = DEFAULT_TEST_STEPS if test_mode else DEFAULT_FULL_STEPS

            yield from log(f"Generating shot {shot_number}: {prompt[:50]}...")

            try:
                frames = generate_shot(pipe, prompt, negative_prompt, num_frames, num_steps, seed)
                out_filename = make_safe_filename(visual_prompt or "untitled", shot_number)
                out_path = os.path.join(out_dir, out_filename)
                export_to_video(frames, out_path, fps=fps)
                generated_files.append(out_path)
                yield from log(f"Saved video: {out_filename}")
            except Exception as e:
                yield from log(f"ERROR on shot {shot_number}: {e}")

        # Create ZIP file
        if not generated_files:
            yield "\n".join(logs), "Finished, but no files were generated.", None
            return
            
        zip_path = os.path.join(out_dir, "documentary_videos.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in generated_files:
                zf.write(f, arcname=os.path.basename(f))
        
        yield from log(f"All done! ZIP file created with {len(generated_files)} video(s).")
        yield "\n".join(logs), "SUCCESS!", zip_path

# --- GRADIO UI LAYOUT ---
with gr.Blocks(title="Robin Hood AI — Documentary Maker") as demo:
    gr.Markdown("# Robin Hood AI — Documentary Maker\nUpload your script as a CSV and turn it into a movie. For free.")
    with gr.Row():
        with gr.Column(scale=1):
            csv_in = gr.File(label="Upload CSV Script", file_types=[".csv"])
            test_mode = gr.Checkbox(value=True, label="Test Mode (fast, 1-second clips)")
            test_shot = gr.Number(value=4, label="Test Shot # (if Test Mode is on)")
            fps_in = gr.Slider(minimum=12, maximum=60, step=1, value=DEFAULT_FPS, label="Video FPS")
            seed_in = gr.Number(value=42, label="Random Seed (keeps style consistent)")
            start_btn = gr.Button("Generate Documentary", variant="primary")
        with gr.Column(scale=2):
            logs_out = gr.Textbox(label="Progress Logs", lines=15, interactive=False)
            status_out = gr.Textbox(label="Overall Status", interactive=False)
            download_out = gr.File(label="Download ZIP of Videos", interactive=False)

    start_btn.click(
        run_generation,
        inputs=[csv_in, test_mode, test_shot, fps_in, seed_in],
        outputs=[logs_out, status_out, download_out]
    )

if __name__ == "__main__":
    demo.launch()