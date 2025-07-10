# PoC Script for Text-to-Video with Audio Generation on Colab TPU
# Features:
# 1. Text-to-Video generation using ModelScopeT2V.
# 2. Text-to-Audio (Speech) generation using Bark.
# 3. Combination of video and audio using moviepy.
# 4. Designed with Colab TPU and memory constraints in mind.
#
# To Run in Colab:
# 1. Ensure your Colab runtime is set to TPU.
# 2. Install dependencies by running the pip install commands from the project plan
#    (torch, torch_xla, diffusers, transformers, accelerate, scipy, moviepy) in a Colab cell.
# 3. Paste this entire script into a new Colab cell.
# 4. Call `run_t2v_audio_poc()` at the end of the cell or in a subsequent cell.
# 5. View the output file paths and the display message for the final video.

import torch
import torch_xla.core.xla_model as xm # For TPU
import diffusers
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from transformers import AutoProcessor, AutoModel, BarkModel
import scipy.io.wavfile
from moviepy.editor import VideoFileClip, AudioFileClip
import os
import gc

# --- Configuration ---
# Model IDs from Hugging Face
T2V_MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"  # Text-to-Video model
BARK_MODEL_ID = "suno/bark"  # Text-to-Audio (Speech) model, large version

# Default prompts and output filenames
SAMPLE_PROMPT = "A panda surfing on a wakeboard"  # Main prompt for video and audio
NEGATIVE_PROMPT = "ugly, blurry, low quality, watermark, text"
VIDEO_FILE_PATH = "generated_video_silent.mp4"
AUDIO_FILE_PATH = "generated_audio.wav"
FINAL_VIDEO_PATH = "generated_video_with_audio.mp4"

VIDEO_DURATION_S = 4 # Shorter for PoC
VIDEO_FPS = 8        # Lower FPS for faster PoC
VIDEO_FRAMES = int(VIDEO_DURATION_S * VIDEO_FPS)
AUDIO_SAMPLING_RATE = 24000 # Bark's default

# --- Helper Functions ---
def get_xla_device():
    """Acquires the XLA device (TPU). Falls back to CPU if TPU is not available."""
    try:
        device = xm.xla_device()
        print(f"Successfully acquired XLA device: {device}")
        return device
    except Exception as e:
        print(f"XLA device not available: {e}. Falling back to CPU (expect very slow performance).")
        return torch.device("cpu")

def cleanup_files(*filenames):
    """Removes specified files if they exist to ensure a clean run."""
    for filename in filenames:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Successfully removed existing file: {filename}")
            except Exception as e:
                print(f"Error removing file {filename}: {e}")

# --- Main PoC Logic ---
def run_t2v_audio_poc():
    """
    Runs the end-to-end Text-to-Video and Text-to-Audio proof of concept.
    1. Initializes TPU device.
    2. Cleans up any previous output files.
    3. Generates a silent video based on SAMPLE_PROMPT.
    4. Generates audio based on SAMPLE_PROMPT.
    5. Combines the video and audio.
    6. Prints paths to output files.
    """
    print("Starting Text-to-Video with Audio PoC...")
    device = get_xla_device()

    # Clean up previous run's files for a fresh execution
    cleanup_files(VIDEO_FILE_PATH, AUDIO_FILE_PATH, FINAL_VIDEO_PATH, 'temp-audio.m4a')

    # Step 1: Text-to-Video Generation (ModelScopeT2V)
    # -------------------------------------------------
    print("\n--- Step 1: Generating Video ---")
    t2v_pipe = None
    try:
        t2v_pipe = DiffusionPipeline.from_pretrained(
            T2V_MODEL_ID,
            torch_dtype=torch.float16, # Use float16 for memory
            variant="fp16"
        )
        t2v_pipe.scheduler = DPMSolverMultistepScheduler.from_config(t2v_pipe.scheduler.config)

        # Offloading can be problematic with XLA if not handled carefully by diffusers for XLA
        # For PoC, let's try without it first if memory allows, or enable if OOM
        # t2v_pipe.enable_model_cpu_offload() # May not work well with XLA directly
        # t2v_pipe.enable_vae_slicing()

        t2v_pipe = t2v_pipe.to(device)
        print(f"T2V Model loaded on {device}")

        video_frames = t2v_pipe(
            SAMPLE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=20, # Fewer steps for PoC
            num_frames=VIDEO_FRAMES,
            guidance_scale=7.5,
            # generator=torch.Generator(device=device).manual_seed(42) # Generator device needs to match pipeline
        ).frames

        export_to_video(video_frames, VIDEO_FILE_PATH, fps=VIDEO_FPS)
        print(f"Silent video saved to {VIDEO_FILE_PATH}")

    except Exception as e:
        print(f"Error during T2V generation: {e}")
    finally:
        if t2v_pipe is not None:
            del t2v_pipe
        gc.collect() # Attempt to free memory
        if str(device) != "cpu": # Specific XLA cleanup might be needed or implicit
             xm.mark_step() # Crucial for XLA: signals end of a computation step, helps manage memory and execution.

    # Step 2: Text-to-Audio Generation (Bark)
    # ----------------------------------------
    print("\n--- Step 2: Generating Audio ---")
    bark_model = None
    bark_processor = None
    try:
        print("Loading Bark processor...")
        bark_processor = AutoProcessor.from_pretrained(BARK_MODEL_ID)
        print("Loading Bark model...")
        # Using BarkModel directly. Ensure torch_dtype for memory saving on compatible devices.
        bark_model = BarkModel.from_pretrained(BARK_MODEL_ID, torch_dtype=torch.float16).to(device)
        # bark_model.enable_model_cpu_offload() # Potentially useful for very large models if XLA allows/benefits

        print(f"Bark Model loaded on {device}")

        # Prepare prompt for Bark. It can handle non-verbal cues like [laughs].
        # Speaker presets (e.g., "v2/en_speaker_6") can be used with `voice_preset` argument in processor.
        # For simplicity, using the direct prompt.
        print(f"Tokenizing audio prompt: '{SAMPLE_PROMPT}'")
        inputs = bark_processor(SAMPLE_PROMPT, return_tensors="pt", voice_preset=None)
        inputs = {k: v.to(device) for k, v in inputs.items()} # Move tokenized inputs to device

        # Generate audio. `do_sample=True` is recommended for more natural sounding audio.
        # `max_length` is a heuristic to ensure enough audio is generated for the desired video duration.
        # Bark generates audio in segments; actual length depends on content and EOS.
        # Temperatures (fine_temperature, coarse_temperature) control randomness/creativity.
        print("Generating audio waveform...")
        with torch.no_grad(): # Inference doesn't require gradients
            speech_values = bark_model.generate(
                **inputs,
                do_sample=True,
                fine_temperature=0.7,
                coarse_temperature=0.7,
                # Heuristic for max_length: try to generate slightly more than needed.
                # Bark's token-to-time is not fixed. This aims to provide enough material.
                max_length=int(AUDIO_SAMPLING_RATE * VIDEO_DURATION_S * 1.5)
            ).cpu().numpy().squeeze() # Move to CPU and convert to NumPy array

        print(f"Saving audio to {AUDIO_FILE_PATH} with sampling rate {AUDIO_SAMPLING_RATE}...")
        scipy.io.wavfile.write(AUDIO_FILE_PATH, rate=AUDIO_SAMPLING_RATE, data=speech_values)
        print(f"Audio successfully saved to {AUDIO_FILE_PATH}")

    except Exception as e:
        print(f"Error during audio generation: {e}")
    finally:
        # Cleanup Bark model components from memory
        if bark_model is not None:
            del bark_model
            print("Bark model unloaded.")
        if bark_processor is not None:
            del bark_processor
            print("Bark processor unloaded.")
        gc.collect() # Python garbage collection
        if str(device) != "cpu":
            xm.mark_step() # Signal XLA step completion

    # Step 3: Combine Video and Audio using moviepy
    # ---------------------------------------------
    print("\n--- Step 3: Combining Video and Audio ---")
    try:
        if os.path.exists(VIDEO_FILE_PATH) and os.path.exists(AUDIO_FILE_PATH):
            print(f"Loading video file: {VIDEO_FILE_PATH}")
            video_clip = VideoFileClip(VIDEO_FILE_PATH)
            print(f"Loading audio file: {AUDIO_FILE_PATH}")
            audio_clip = AudioFileClip(AUDIO_FILE_PATH)

            # Adjust audio duration to match video duration
            # If audio is longer, it's trimmed. If shorter, it's padded with silence.
            print("Adjusting audio duration to match video...")
            if audio_clip.duration > video_clip.duration:
                print(f"Audio duration ({audio_clip.duration:.2f}s) longer than video ({video_clip.duration:.2f}s). Trimming audio.")
                audio_clip = audio_clip.subclip(0, video_clip.duration)
            elif audio_clip.duration < video_clip.duration:
                print(f"Audio duration ({audio_clip.duration:.2f}s) shorter than video ({video_clip.duration:.2f}s). Padding audio with silence.")
                silence_duration = video_clip.duration - audio_clip.duration
                if silence_duration > 0.01: # Avoid padding for negligible differences
                    from moviepy.editor import CompositeAudioClip
                    import numpy as np

                    num_audio_channels = audio_clip.nchannels
                    # Ensure correct shape for silence array based on channels
                    silence_array_shape = (int(silence_duration * AUDIO_SAMPLING_RATE), num_audio_channels) if num_audio_channels > 1 else (int(silence_duration * AUDIO_SAMPLING_RATE),)
                    silence_array = np.zeros(silence_array_shape, dtype=np.float32) # Use float32 for moviepy

                    audio_fps = getattr(audio_clip, 'fps', AUDIO_SAMPLING_RATE)
                    if audio_fps is None: audio_fps = AUDIO_SAMPLING_RATE

                    # Use moviepy.audio.AudioClip.AudioArrayClip for creating an audio clip from numpy array
                    silence_clip = moviepy.audio.AudioClip.AudioArrayClip(silence_array, fps=audio_fps)
                    audio_clip = CompositeAudioClip([audio_clip, silence_clip.set_start(audio_clip.duration)])

                # Ensure the final audio_clip has the exact video duration
                audio_clip = audio_clip.set_duration(video_clip.duration)

            print("Setting audio for the video clip...")
            final_clip = video_clip.set_audio(audio_clip)

            print(f"Writing final video with audio to {FINAL_VIDEO_PATH}...")
            # Specify codec for video and audio to ensure compatibility
            final_clip.write_videofile(
                FINAL_VIDEO_PATH,
                codec="libx264",          # Common video codec
                audio_codec="aac",        # Common audio codec
                temp_audiofile='temp-audio.m4a', # Temporary file for audio processing
                remove_temp=True          # Clean up temporary file
            )
            print(f"Final video with audio successfully saved to {FINAL_VIDEO_PATH}")

            # In Colab, you would typically display the video like this:
            # from IPython.display import Video
            # Video(FINAL_VIDEO_PATH, embed=True, html_attributes="controls autoplay loop")
            print(f"\nTo display video in Colab, use: from IPython.display import Video; Video('{FINAL_VIDEO_PATH}', embed=True)")

        else:
            print("Skipping combination: Silent video or audio file is missing.")
    except Exception as e:
        print(f"Error during video and audio combination: {e}")
    finally:
        # Cleanup moviepy clips
        if 'video_clip' in locals() and video_clip: video_clip.close()
        if 'audio_clip' in locals() and audio_clip: audio_clip.close()
        if 'final_clip' in locals() and final_clip: final_clip.close()
        if 'silence_clip' in locals() and silence_clip: silence_clip.close()
        gc.collect()

    print("\nText-to-Video with Audio PoC finished.")

if __name__ == "__main__":
    # This block is for potential direct script execution (e.g., local testing without Colab)
    # In Colab, you would typically call run_t2v_audio_poc() directly in a cell.
    print("PoC Script Main Block Executed.")
    print("To run the PoC: Call the function run_t2v_audio_poc()")
    # Example: run_t2v_audio_poc() # Uncomment to run if executing script directly (and if environment is set up)
    pass
