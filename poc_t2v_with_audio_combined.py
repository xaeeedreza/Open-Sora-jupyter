# PoC Script for Text-to-Video with Combined Audio
# Step 5: Combine Video and Audio
# This script generates video, then audio, then combines them.

import torch
import torch_xla.core.xla_model as xm
import diffusers
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import moviepy.audio.AudioClip # For AudioArrayClip
import numpy as np # For creating silence array
import os
import gc

# --- Configuration ---
T2V_MODEL_ID = "cerspense/zeroscope_v2_576w"
BARK_MODEL_ID = "suno/bark-small"

SAMPLE_PROMPT = "A cute puppy playing in a field of flowers, gentle breeze"
NEGATIVE_PROMPT = "ugly, blurry, low quality, watermark, text, deformed"

# File Paths
SILENT_VIDEO_FILE_PATH = "generated_silent_video_temp.mp4"
AUDIO_FILE_PATH = "generated_audio_temp.wav"
FINAL_VIDEO_WITH_AUDIO_PATH = "generated_video_with_audio_final.mp4"
MOVIEPY_TEMP_AUDIO_FILE = "moviepy_temp_audio.m4a" # for moviepy intermediate

# Video PoC parameters
VIDEO_DURATION_S = 2
VIDEO_FPS = 12
NUM_FRAMES = int(VIDEO_DURATION_S * VIDEO_FPS)
T2V_NUM_INFERENCE_STEPS = 20
T2V_GUIDANCE_SCALE = 9.0

# Audio PoC parameters
AUDIO_SAMPLING_RATE = 24000
BARK_MAX_NEW_TOKENS_HEURISTIC = int(75 * VIDEO_DURATION_S * 1.5) # Approx 75 tokens/sec for Bark, with 1.5x buffer
BARK_MAX_NEW_TOKENS_CAP = 768 # Max cap for Bark tokens

# --- Helper Functions ---
def get_xla_device_or_cpu():
    try:
        device = xm.xla_device()
        print(f"Successfully acquired XLA device: {device}")
        return device
    except Exception as e:
        print(f"XLA device not available: {e}. Falling back to CPU (expect very slow performance).")
        return torch.device("cpu")

def cleanup_files_if_exist(*filenames):
    for filename in filenames:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Successfully removed existing file: {filename}")
            except Exception as e:
                print(f"Error removing file {filename}: {e}")

# --- Main T2V, Audio, and Combination Logic ---
def generate_and_combine_video_audio():
    print(f"Starting Text-to-Video with Audio Combination PoC...")
    device = get_xla_device_or_cpu()

    # Clean up all potential output and temporary files from previous runs
    cleanup_files_if_exist(SILENT_VIDEO_FILE_PATH, AUDIO_FILE_PATH, FINAL_VIDEO_WITH_AUDIO_PATH, MOVIEPY_TEMP_AUDIO_FILE)

    video_generated_successfully = False
    audio_generated_successfully = False

    # 1. Text-to-Video (Silent - ZeroScope)
    # --------------------------------------
    print(f"\n--- Generating Silent Video (Model: {T2V_MODEL_ID}) ---")
    t2v_pipe = None
    try:
        print(f"Loading T2V pipeline for {T2V_MODEL_ID}...")
        t2v_pipe = DiffusionPipeline.from_pretrained(T2V_MODEL_ID, torch_dtype=torch.float16)
        t2v_pipe.scheduler = DPMSolverMultistepScheduler.from_config(t2v_pipe.scheduler.config)
        print(f"Moving T2V pipeline to device: {device}...")
        t2v_pipe = t2v_pipe.to(device)
        if str(device) != "cpu": xm.mark_step()
        print("T2V pipeline successfully loaded and moved to device.")

        print(f"Generating video frames for prompt: '{SAMPLE_PROMPT}'...")
        video_frames_tensor = t2v_pipe(
            prompt=SAMPLE_PROMPT, negative_prompt=NEGATIVE_PROMPT,
            num_frames=NUM_FRAMES, num_inference_steps=T2V_NUM_INFERENCE_STEPS,
            guidance_scale=T2V_GUIDANCE_SCALE, height=320, width=576,
        ).frames
        if str(device) != "cpu": xm.mark_step()
        print("Video frames generated.")

        print(f"Exporting frames to video: {SILENT_VIDEO_FILE_PATH} at {VIDEO_FPS} FPS...")
        export_to_video(video_frames_tensor, SILENT_VIDEO_FILE_PATH, fps=VIDEO_FPS)
        print(f"Silent video saved to {SILENT_VIDEO_FILE_PATH}")
        video_generated_successfully = True
    except Exception as e:
        print(f"An error occurred during silent T2V generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if t2v_pipe is not None: del t2v_pipe; print("T2V pipeline unloaded.")
        gc.collect()
        if str(device) != "cpu": xm.mark_step()

    # 2. Text-to-Audio (Bark)
    # -------------------------
    if video_generated_successfully: # Only proceed if video was created
        print(f"\n--- Generating Audio (Model: {BARK_MODEL_ID}) ---")
        bark_processor = None
        bark_model = None
        try:
            print(f"Loading Bark processor for {BARK_MODEL_ID}...")
            bark_processor = AutoProcessor.from_pretrained(BARK_MODEL_ID)
            print(f"Loading Bark model {BARK_MODEL_ID}...")
            bark_model = BarkModel.from_pretrained(BARK_MODEL_ID, torch_dtype=torch.float16)
            print(f"Moving Bark model to device: {device}...")
            bark_model = bark_model.to(device)
            if str(device) != "cpu": xm.mark_step()
            print("Bark model successfully loaded and moved to device.")

            print(f"Tokenizing audio prompt: '{SAMPLE_PROMPT}'...")
            inputs = bark_processor(SAMPLE_PROMPT, return_tensors="pt", voice_preset=None)
            inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
            if str(device) != "cpu": xm.mark_step()

            print("Generating audio waveform...")
            actual_max_new_tokens = min(BARK_MAX_NEW_TOKENS_CAP, BARK_MAX_NEW_TOKENS_HEURISTIC)
            with torch.no_grad():
                speech_values = bark_model.generate(
                    **inputs_on_device, do_sample=True, fine_temperature=0.7, coarse_temperature=0.7,
                    max_new_tokens=actual_max_new_tokens
                ).cpu().numpy().squeeze()
            if str(device) != "cpu": xm.mark_step()
            print("Audio waveform generated.")

            print(f"Saving audio to {AUDIO_FILE_PATH} at {AUDIO_SAMPLING_RATE} Hz...")
            scipy.io.wavfile.write(AUDIO_FILE_PATH, rate=AUDIO_SAMPLING_RATE, data=speech_values.astype(np.float32)) # Ensure float32 for scipy
            print(f"Audio successfully saved to {AUDIO_FILE_PATH}")
            audio_generated_successfully = True
        except Exception as e:
            print(f"An error occurred during audio generation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if bark_model is not None: del bark_model; print("Bark model unloaded.")
            if bark_processor is not None: del bark_processor; print("Bark processor unloaded.")
            gc.collect()
            if str(device) != "cpu": xm.mark_step()
    else:
        print("Skipping audio generation because video generation failed.")

    # 3. Combine Video and Audio using moviepy
    # ------------------------------------------
    if video_generated_successfully and audio_generated_successfully:
        print("\n--- Combining Video and Audio ---")
        video_clip_obj, audio_clip_obj, final_clip_obj, silence_clip_obj = None, None, None, None # For finally block
        try:
            print(f"Loading video file: {SILENT_VIDEO_FILE_PATH}")
            video_clip_obj = VideoFileClip(SILENT_VIDEO_FILE_PATH)
            print(f"Loading audio file: {AUDIO_FILE_PATH}")
            audio_clip_obj = AudioFileClip(AUDIO_FILE_PATH)

            print("Adjusting audio duration to match video...")
            if audio_clip_obj.duration > video_clip_obj.duration:
                print(f"Audio duration ({audio_clip_obj.duration:.2f}s) longer than video ({video_clip_obj.duration:.2f}s). Trimming audio.")
                audio_clip_obj = audio_clip_obj.subclip(0, video_clip_obj.duration)
            elif audio_clip_obj.duration < video_clip_obj.duration:
                silence_duration = video_clip_obj.duration - audio_clip_obj.duration
                print(f"Audio duration ({audio_clip_obj.duration:.2f}s) shorter than video ({video_clip_obj.duration:.2f}s). Padding with {silence_duration:.2f}s of silence.")
                if silence_duration > 0.01:
                    num_audio_channels = audio_clip_obj.nchannels
                    silence_array_shape = (int(silence_duration * AUDIO_SAMPLING_RATE), num_audio_channels) if num_audio_channels > 1 else (int(silence_duration * AUDIO_SAMPLING_RATE),)
                    silence_array = np.zeros(silence_array_shape, dtype=np.float32)

                    audio_fps = getattr(audio_clip_obj, 'fps', AUDIO_SAMPLING_RATE)
                    if audio_fps is None: audio_fps = AUDIO_SAMPLING_RATE

                    silence_clip_obj = moviepy.audio.AudioClip.AudioArrayClip(silence_array, fps=audio_fps)
                    audio_clip_obj = CompositeAudioClip([audio_clip_obj, silence_clip_obj.set_start(audio_clip_obj.duration)])
                audio_clip_obj = audio_clip_obj.set_duration(video_clip_obj.duration)

            print("Setting audio for the video clip...")
            final_clip_obj = video_clip_obj.set_audio(audio_clip_obj)

            print(f"Writing final video with audio to {FINAL_VIDEO_WITH_AUDIO_PATH}...")
            final_clip_obj.write_videofile(
                FINAL_VIDEO_WITH_AUDIO_PATH, codec="libx264", audio_codec="aac",
                temp_audiofile=MOVIEPY_TEMP_AUDIO_FILE, remove_temp=True,
                threads=2 # Limit threads for Colab stability potentially
            )
            print(f"Final video with audio successfully saved to {FINAL_VIDEO_WITH_AUDIO_PATH}")
            print(f"\nTo display video in Colab, use: from IPython.display import Video; Video('{FINAL_VIDEO_WITH_AUDIO_PATH}', embed=True)")
        except Exception as e:
            print(f"An error occurred during video and audio combination: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if video_clip_obj: video_clip_obj.close()
            if audio_clip_obj: audio_clip_obj.close()
            if final_clip_obj: final_clip_obj.close()
            if silence_clip_obj: silence_clip_obj.close() # Ensure silence_clip is also closed if created
            gc.collect()
    else:
        print("Skipping video/audio combination because one or both generation steps failed.")

    print("\nText-to-Video with Audio Combination PoC finished.")

if __name__ == "__main__":
    print("Script for T2V, Bark audio, and moviepy combination (poc_t2v_with_audio_combined.py).")
    print("To run: Ensure TPU setup, install deps, then call generate_and_combine_video_audio()")
    # generate_and_combine_video_audio() # Uncomment to run
    pass
