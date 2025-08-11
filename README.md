# RobinHoodAI
Free AI tool to make documentaries—upload a CSV to create!

A free, open-source tool to automatically create documentary-style videos from a simple script. This project takes a CSV file containing narration and visual prompts and generates a complete movie.

## What It Does

This project is a three-part production pipeline:

1.  **`generate_csv_video.py`**: Reads the CSV and uses a text-to-video model to generate silent video clips for each shot.
2.  **`run_narration_generation.py`**: Reads the same CSV and uses a text-to-speech model to generate narration audio for each shot.
3.  **`imagine_movie_maker.py`**: Assembles the final movie by stitching the video and audio clips together in the correct order.

## How to Use (Local Setup)

This project requires two separate, isolated Python environments due to conflicting library dependencies. The recommended setup is using **Conda**.

### Step 1: Create the Video Environment

This environment uses an older, stable version of PyTorch for the video model.

```bash
# Create and activate the environment
conda create --name video_env python=3.11 -y
conda activate video_env

# Install dependencies
pip install "numpy<2"
pip install torch==2.4.0+cu118 torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu118
pip install diffusers pandas accelerate imageio imageio-ffmpeg

Step 2: Create the Audio Environment
This environment uses a modern version of PyTorch for the text-to-speech model.
code
Bash
# Create and activate the environment
conda create --name audio_env python=3.11 -y
conda activate audio_env

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install speechbrain soundfile pandas datasets sentencepiece

Step 3: Run the Production Pipeline
Execute the scripts in the correct order, using the correct environment for each.
code
Bash
# First, generate all the audio clips
conda activate audio_env
python run_narration_generation.py

# Second, generate all the video clips (this will take a very long time)
conda activate video_env
python generate_csv_video.py

# Finally, assemble the movie (this uses your main 'base' environment)
conda deactivate
pip install moviepy # (Install moviepy in your base env once)
python imagine_movie_maker.py

The Future: Hugging Face App
The app.py file contains the code for a Gradio web application that wraps this entire process. The goal is to deploy this to a Hugging Face Space to make it accessible to everyone.
Credits
This project was a collaboration between a human visionary and a team of AI assistants, including Gemini, ChatGPT, and Grok.
