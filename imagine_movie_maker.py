# File: imagine_movie_maker.py
# Purpose: Assembles the final documentary with professional transitions.
# To Run: Activate the base conda environment: `conda activate base`
#         Then run: `python imagine_movie_maker.py`

import pandas as pd
import os
from moviepy.editor import * # Import everything for advanced editing
from tqdm import tqdm # For a beautiful progress bar

print("--- Imagine Movie Maker (Director's Cut): The Final Assembly ---")

# --- CONFIGURATION ---
video_clips_dir = r"C:\Users\eazy8\Downloads\backup chatgbt\video\GAIA_Output"
audio_clips_dir = r"C:\Users\eazy8\Downloads\backup chatgbt\video\GAIA_Output\narration"
csv_path = r"C:\Users\eazy8\Downloads\backup chatgbt\video\GAIA_Production_Master_Sheet.csv"
final_movie_path = r"C:\Users\eazy8\Downloads\backup chatgbt\video\GAIA_The_Movie_Directors_Cut.mp4"

# --- NEW FEATURE: Professional Transitions ---
# How long should the fade between clips be, in seconds?
TRANSITION_DURATION = 0.5 # A half-second crossfade

# --- HELPER FUNCTION ---
def make_safe_filename(prompt, shot_number, extension):
    words = prompt.split()[:4]
    safe_name = "_".join(word.lower().strip(".,!?") for word in words)
    return f"shot_{shot_number:03d}_{safe_name}.{extension}"

def main():
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"FATAL: Error loading CSV: {str(e)}")
        return

    print("\n--- Starting the Edit: Synchronizing Clips ---")
    
    synchronized_clips = []
    
    # Use tqdm for a progress bar during the first phase
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Syncing Clips"):
        shot_number = row['Shot#']
        visual_prompt = row['Visual Prompt']

        video_filename = make_safe_filename(visual_prompt, shot_number, "mp4")
        audio_filename = f"shot_{shot_number:03d}.wav"
        
        video_path = os.path.join(video_clips_dir, video_filename)
        audio_path = os.path.join(audio_clips_dir, audio_filename)

        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            continue

        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            # Trim or loop video to perfectly match the audio duration
            if video_clip.duration < audio_clip.duration:
                video_clip = video_clip.loop(duration=audio_clip.duration)
            else:
                video_clip = video_clip.subclip(0, audio_clip.duration)

            # Attach the audio to the video
            final_clip = video_clip.set_audio(audio_clip)
            synchronized_clips.append(final_clip)
        except Exception as e:
            print(f"!!! ERROR synchronizing Shot #{shot_number}: {e}")
            continue

    if not synchronized_clips:
        print("\nFATAL: No valid clips to assemble. Exiting.")
        return

    # --- NEW FEATURE: Apply Crossfade Transitions ---
    print(f"\n--- Assembling Final Movie with {TRANSITION_DURATION}s crossfades ---")
    
    # Use the advanced `concatenate_videoclips` with a transition effect
    final_movie = concatenate_videoclips(synchronized_clips, transition=vfx.fadein, transition_duration=TRANSITION_DURATION)
    
    print("Writing final movie file. This may take several minutes...")
    try:
        final_movie.write_videofile(
            final_movie_path, 
            codec='libx264', 
            audio_codec='aac'
        )
        
        print("\n------------------------------------------")
        print(f"THE MOVIE IS COMPLETE! Saved at: {final_movie_path}")
        print("------------------------------------------")
        
        # --- NEW FEATURE: Instant Gratification ---
        # Automatically open the movie for you when it's done!
        try:
            print("Attempting to open the movie for you...")
            os.startfile(final_movie_path)
        except AttributeError:
            print("(Auto-opening files is only supported on Windows.)")

    except Exception as e:
        print(f"\n!!! FATAL ERROR during final render: {e}")

if __name__ == "__main__":
    main()