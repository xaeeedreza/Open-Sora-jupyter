# PoC Script for Text-to-Video with Combined Audio
# Generates:
# 1. Silent video using a Text-to-Video model (ZeroScope).
# 2. Audio (speech) using a Text-to-Audio model (Bark).
# 3. Combines the video and audio into a single MP4 file using moviepy.
#
# Designed for Google Colab with TPU acceleration.
#
# To Run in Colab:
# 1. Ensure your Colab runtime is set to TPU (Runtime -> Change runtime type -> TPU).
# 2. Install dependencies in a cell:
#    !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#    # Use the correct torch_xla install command for your Colab environment! Example:
#    !pip install torch_xla[tpu]~=2.0.0 -f https://storage.googleapis.com/pytorch-xla-releases/index.html
#    !pip install diffusers transformers accelerate scipy moviepy --quiet
# 3. Paste this entire script into a new Colab cell.
# 4. Call `generate_and_combine_video_audio()` at the end of the cell or in a subsequent cell.
# 5. Monitor the output for progress and the final file path.

import torch
import torch_xla.core.xla_model as xm # For TPU operations
import diffusers
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from transformers import AutoProcessor, BarkModel # Using BarkModel directly for better control
import scipy.io.wavfile
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import moviepy.audio.AudioClip # For AudioArrayClip
import numpy as np
import os
import gc # Garbage collector

# --- Configuration ---
# Model IDs from Hugging Face
T2V_MODEL_ID = "cerspense/zeroscope_v2_576w"    # Text-to-Video model (ZeroScope, 576w variant)
BARK_MODEL_ID = "suno/bark-small"              # Text-to-Audio (Speech) model (Bark, small variant for Colab)

# Prompts & Output Files
SAMPLE_PROMPT = "A joyful robot dancing in a futuristic city street, neon lights reflecting" # Used for both video & audio
NEGATIVE_PROMPT = "ugly, blurry, low quality, watermark, text, deformed, human, static" # Common negative prompts
SILENT_VIDEO_FILE_PATH = "generated_silent_video_temp.mp4"  # Temporary silent video
AUDIO_FILE_PATH = "generated_audio_temp.wav"                # Temporary audio file
FINAL_VIDEO_WITH_AUDIO_PATH = "generated_video_with_audio_final.mp4" # Final output
MOVIEPY_TEMP_AUDIO_FILE = "moviepy_temp_audio.m4a"         # moviepy's intermediate audio file for muxing

# Video Generation Parameters
VIDEO_DURATION_S = 3  # Duration of the video in seconds
VIDEO_FPS = 10        # Frames per second for the video
NUM_FRAMES = int(VIDEO_DURATION_S * VIDEO_FPS) # Total frames to generate
T2V_HEIGHT = 320      # Height for ZeroScope model (ensure multiple of VAE factor, e.g., 8 or 16)
T2V_WIDTH = 576       # Width for ZeroScope model (ensure multiple of VAE factor)
T2V_NUM_INFERENCE_STEPS = 25 # Number of denoising steps for T2V (higher = better quality, slower)
T2V_GUIDANCE_SCALE = 9.0   # How strongly the video generation adheres to the prompt

# Audio Generation Parameters
AUDIO_SAMPLING_RATE = 24000  # Bark's default output sampling rate (Hz)
# Heuristic for Bark's max_new_tokens: approx 75 tokens/sec for Bark, with a 1.5x buffer.
# This helps ensure enough audio material is generated for the desired duration.
# Bark often stops early via EOS, so this is more of an upper limit.
BARK_MAX_NEW_TOKENS_HEURISTIC = int(75 * VIDEO_DURATION_S * 1.5)
BARK_MAX_NEW_TOKENS_CAP = 768  # Absolute max cap for Bark tokens to prevent excessive generation

# --- Helper Functions ---
def get_xla_device_or_cpu():
    """Acquires the XLA device (TPU). Falls back to CPU if TPU is not available."""
    try:
        device = xm.xla_device()
        print(f"Successfully acquired XLA device: {device}")
        return device
    except Exception as e:
        print(f"XLA device not available: {e}. Falling back to CPU (expect very slow performance).")
        return torch.device("cpu")

def cleanup_files_if_exist(*filenames):
    """Removes specified files if they exist to ensure a clean run."""
    for filename in filenames:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Successfully removed existing file: {filename}")
            except Exception as e:
                print(f"Error removing file {filename}: {e}")

# --- Main T2V, Audio, and Combination Logic ---
def generate_and_combine_video_audio():
    """
    Main function to run the end-to-end Text-to-Video with Audio generation and combination PoC.
    Steps:
    1. Initialize TPU device.
    2. Clean up previous output files.
    3. Generate a silent video using ZeroScope.
    4. Generate speech audio using Bark.
    5. Combine the video and audio using moviepy.
    """
    print(f"Starting Text-to-Video with Audio Combination PoC (v{VERSION_INFO})...") # Added version
    device = get_xla_device_or_cpu()

    cleanup_files_if_exist(
        SILENT_VIDEO_FILE_PATH, AUDIO_FILE_PATH,
        FINAL_VIDEO_WITH_AUDIO_PATH, MOVIEPY_TEMP_AUDIO_FILE
    )

    video_generated_successfully = False
    audio_generated_successfully = False

    # === Step 1: Text-to-Video (Silent - ZeroScope) ===
    print(f"\n--- Generating Silent Video (Model: {T2V_MODEL_ID}) ---")
    t2v_pipe = None
    try:
        # Load T2V pipeline
        print(f"Loading T2V pipeline for {T2V_MODEL_ID} with float16 precision...")
        t2v_pipe = DiffusionPipeline.from_pretrained(T2V_MODEL_ID, torch_dtype=torch.float16)
        t2v_pipe.scheduler = DPMSolverMultistepScheduler.from_config(t2v_pipe.scheduler.config)

        # Move to XLA device
        print(f"Moving T2V pipeline to device: {device} (This may take a moment for XLA compilation on first run)...")
        t2v_pipe = t2v_pipe.to(device)
        if str(device) != "cpu": xm.mark_step() # Important after model transfer to TPU
        print("T2V pipeline successfully loaded and moved to device.")

        # Generate frames
        print(f"Generating {NUM_FRAMES} video frames for prompt: '{SAMPLE_PROMPT}'...")
        print(f"Using {T2V_NUM_INFERENCE_STEPS} steps, guidance scale {T2V_GUIDANCE_SCALE}, HxW: {T2V_HEIGHT}x{T2V_WIDTH}")
        video_frames_tensor = t2v_pipe(
            prompt=SAMPLE_PROMPT, negative_prompt=NEGATIVE_PROMPT,
            num_frames=NUM_FRAMES, num_inference_steps=T2V_NUM_INFERENCE_STEPS,
            guidance_scale=T2V_GUIDANCE_SCALE, height=T2V_HEIGHT, width=T2V_WIDTH,
        ).frames
        if str(device) != "cpu": xm.mark_step() # After inference
        print("Video frames generated.")

        # Export to MP4
        print(f"Exporting frames to video: {SILENT_VIDEO_FILE_PATH} at {VIDEO_FPS} FPS...")
        export_to_video(video_frames_tensor, SILENT_VIDEO_FILE_PATH, fps=VIDEO_FPS)
        print(f"Silent video saved to {SILENT_VIDEO_FILE_PATH}")
        video_generated_successfully = True
    except Exception as e:
        print(f"An error occurred during silent T2V generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release T2V model from memory
        if t2v_pipe is not None:
            del t2v_pipe
            print("T2V pipeline unloaded from memory.")
        gc.collect() # Trigger garbage collection
        if str(device) != "cpu": xm.mark_step() # Ensure XLA operations are flushed / resources potentially freed

    # === Step 2: Text-to-Audio (Bark) ===
    if video_generated_successfully:
        print(f"\n--- Generating Audio (Model: {BARK_MODEL_ID}) ---")
        bark_processor = None
        bark_model = None
        try:
            # Load Bark processor and model
            print(f"Loading Bark processor for {BARK_MODEL_ID}...")
            bark_processor = AutoProcessor.from_pretrained(BARK_MODEL_ID)
            print(f"Loading Bark model {BARK_MODEL_ID} with float16 precision...")
            bark_model = BarkModel.from_pretrained(BARK_MODEL_ID, torch_dtype=torch.float16)

            # Move to XLA device
            print(f"Moving Bark model to device: {device} (This may take a moment for XLA compilation on first run)...")
            bark_model = bark_model.to(device)
            if str(device) != "cpu": xm.mark_step()
            print("Bark model successfully loaded and moved to device.")

            # Prepare inputs
            print(f"Tokenizing audio prompt: '{SAMPLE_PROMPT}'...")
            inputs = bark_processor(SAMPLE_PROMPT, return_tensors="pt", voice_preset=None) # Using default voice
            inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
            if str(device) != "cpu": xm.mark_step() # After moving inputs

            # Generate audio waveform
            actual_max_new_tokens = min(BARK_MAX_NEW_TOKENS_CAP, BARK_MAX_NEW_TOKENS_HEURISTIC)
            print(f"Generating audio waveform with max_new_tokens={actual_max_new_tokens}...")
            with torch.no_grad(): # Inference mode
                speech_values = bark_model.generate(
                    **inputs_on_device,
                    do_sample=True, fine_temperature=0.7, coarse_temperature=0.7, # Common sampling params
                    max_new_tokens=actual_max_new_tokens
                ).cpu().numpy().squeeze() # To CPU and NumPy array
            if str(device) != "cpu": xm.mark_step()
            print("Audio waveform generated.")

            # Save audio to WAV
            print(f"Saving audio to {AUDIO_FILE_PATH} at {AUDIO_SAMPLING_RATE} Hz...")
            # Ensure data is float32 for scipy, as Bark output is float
            scipy.io.wavfile.write(AUDIO_FILE_PATH, rate=AUDIO_SAMPLING_RATE, data=speech_values.astype(np.float32))
            print(f"Audio successfully saved to {AUDIO_FILE_PATH}")
            audio_generated_successfully = True
        except Exception as e:
            print(f"An error occurred during audio generation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Release Bark model from memory
            if bark_model is not None:
                del bark_model
                print("Bark model unloaded from memory.")
            if bark_processor is not None:
                del bark_processor
                print("Bark processor unloaded from memory.")
            gc.collect()
            if str(device) != "cpu": xm.mark_step()
    else:
        print("\nSkipping audio generation because video generation failed or was skipped.")

    # === Step 3: Combine Video and Audio using moviepy ===
    if video_generated_successfully and audio_generated_successfully:
        print("\n--- Combining Video and Audio using moviepy ---")
        video_clip_obj, audio_clip_obj, final_clip_obj, silence_clip_obj = None, None, None, None
        try:
            # Load video and audio clips
            print(f"Loading silent video file: {SILENT_VIDEO_FILE_PATH}")
            video_clip_obj = VideoFileClip(SILENT_VIDEO_FILE_PATH)
            print(f"Loading audio file: {AUDIO_FILE_PATH}")
            audio_clip_obj = AudioFileClip(AUDIO_FILE_PATH)

            # Adjust audio duration to match video duration
            print(f"Video duration: {video_clip_obj.duration:.2f}s, Original audio duration: {audio_clip_obj.duration:.2f}s")
            if audio_clip_obj.duration > video_clip_obj.duration:
                print("Audio is longer than video. Trimming audio...")
                audio_clip_obj = audio_clip_obj.subclip(0, video_clip_obj.duration)
            elif audio_clip_obj.duration < video_clip_obj.duration:
                silence_duration = video_clip_obj.duration - audio_clip_obj.duration
                print(f"Audio is shorter than video. Padding with {silence_duration:.2f}s of silence...")
                if silence_duration > 0.01: # Only pad if difference is significant
                    num_audio_channels = audio_clip_obj.nchannels
                    silence_array_shape = (int(silence_duration * AUDIO_SAMPLING_RATE), num_audio_channels) if num_audio_channels > 1 else (int(silence_duration * AUDIO_SAMPLING_RATE),)
                    silence_array = np.zeros(silence_array_shape, dtype=np.float32)

                    audio_fps_for_silence = getattr(audio_clip_obj, 'fps', AUDIO_SAMPLING_RATE)
                    if audio_fps_for_silence is None: audio_fps_for_silence = AUDIO_SAMPLING_RATE

                    silence_clip_obj = moviepy.audio.AudioClip.AudioArrayClip(silence_array, fps=audio_fps_for_silence)
                    audio_clip_obj = CompositeAudioClip([audio_clip_obj, silence_clip_obj.set_start(audio_clip_obj.duration)])
                # Ensure final audio duration precisely matches video after potential padding/composition
                audio_clip_obj = audio_clip_obj.set_duration(video_clip_obj.duration)
            print(f"Adjusted audio duration: {audio_clip_obj.duration:.2f}s")

            # Set video's audio to the (adjusted) audio clip
            print("Assigning audio to video clip...")
            final_clip_obj = video_clip_obj.set_audio(audio_clip_obj)

            # Write the final video file
            print(f"Writing final combined video to {FINAL_VIDEO_WITH_AUDIO_PATH}...")
            final_clip_obj.write_videofile(
                FINAL_VIDEO_WITH_AUDIO_PATH,
                codec="libx264",          # Standard video codec
                audio_codec="aac",        # Standard audio codec
                temp_audiofile=MOVIEPY_TEMP_AUDIO_FILE, # For moviepy's internal audio processing
                remove_temp=True,         # Clean up the temporary audio file
                threads=2,                # Limit threads for stability in Colab
                logger='bar'              # Show moviepy progress bar
            )
            print(f"Final video with audio successfully saved to {FINAL_VIDEO_WITH_AUDIO_PATH}")

            # Instructions for displaying in Colab
            print(f"\nSUCCESS! To display the final video in Colab, create a new cell and run:")
            print(f"from IPython.display import Video")
            print(f"Video('{FINAL_VIDEO_WITH_AUDIO_PATH}', embed=True, html_attributes='controls autoplay loop')")

        except Exception as e:
            print(f"An error occurred during video and audio combination: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Close all moviepy clips to release file handles
            if video_clip_obj: video_clip_obj.close(); print("Closed video_clip_obj.")
            if audio_clip_obj: audio_clip_obj.close(); print("Closed audio_clip_obj.")
            if final_clip_obj: final_clip_obj.close(); print("Closed final_clip_obj.")
            if silence_clip_obj: silence_clip_obj.close(); print("Closed silence_clip_obj.")
            gc.collect()
    else:
        print("\nSkipping video/audio combination because one or both of the generation steps failed or were skipped.")

    print("\nText-to-Video with Audio Combination PoC execution finished.")

VERSION_INFO = "1.0.0_doc" # Simple versioning for the script

if __name__ == "__main__":
    print(f"Script: poc_t2v_with_audio_combined_documented.py (Version: {VERSION_INFO})")
    print("This script generates a video with audio from text.")
    print("To run in Colab: Ensure TPU is set up, all dependencies are installed, then call generate_and_combine_video_audio() in a cell.")
    # Example Call (uncomment in Colab or if running script directly after setup):
    # generate_and_combine_video_audio()
    pass
