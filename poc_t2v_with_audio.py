# PoC Script for Text-to-Video with separate Audio Generation
# Step 4: Implement Audio Generation (Bark) - This script generates both video and audio, but doesn't combine yet.
# Based on poc_silent_t2v.py

import torch
import torch_xla.core.xla_model as xm
import diffusers
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from transformers import AutoProcessor, BarkModel # Changed from AutoModel for Bark
import scipy.io.wavfile
import os
import gc

# --- Configuration ---
T2V_MODEL_ID = "cerspense/zeroscope_v2_576w"
BARK_MODEL_ID = "suno/bark-small" # Using bark-small for better Colab compatibility initially

SAMPLE_PROMPT = "A cute puppy playing in a field of flowers, gentle breeze" # Slightly more descriptive for audio
NEGATIVE_PROMPT = "ugly, blurry, low quality, watermark, text, deformed"
SILENT_VIDEO_FILE_PATH = "generated_silent_video_for_audio.mp4" # Renamed for clarity
AUDIO_FILE_PATH = "generated_audio_bark.wav"

# Video PoC parameters
VIDEO_DURATION_S = 2
VIDEO_FPS = 12
NUM_FRAMES = int(VIDEO_DURATION_S * VIDEO_FPS)
T2V_NUM_INFERENCE_STEPS = 20
T2V_GUIDANCE_SCALE = 9.0

# Audio PoC parameters
AUDIO_SAMPLING_RATE = 24000  # Bark's default output sampling rate is 24kHz
BARK_MAX_LENGTH_HEURISTIC_MULTIPLIER = 1.5 # Multiplier for estimating max_length for Bark

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

# --- Main T2V and Audio Generation Logic ---
def generate_video_and_audio_separately():
    print(f"Starting Text-to-Video (silent) and Text-to-Audio (Bark) generation...")
    device = get_xla_device_or_cpu()
    cleanup_files_if_exist(SILENT_VIDEO_FILE_PATH, AUDIO_FILE_PATH)

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
    except Exception as e:
        print(f"An error occurred during silent T2V generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if t2v_pipe is not None:
            del t2v_pipe
            print("T2V pipeline unloaded.")
        gc.collect()
        if str(device) != "cpu": xm.mark_step() # Ensure step marked after potential full block execution

    # 2. Text-to-Audio (Bark)
    # -------------------------
    print(f"\n--- Generating Audio (Model: {BARK_MODEL_ID}) ---")
    bark_processor = None
    bark_model = None
    try:
        print(f"Loading Bark processor for {BARK_MODEL_ID}...")
        bark_processor = AutoProcessor.from_pretrained(BARK_MODEL_ID)
        print(f"Loading Bark model {BARK_MODEL_ID}...")
        # Using BarkModel explicitly and float16 for memory
        bark_model = BarkModel.from_pretrained(BARK_MODEL_ID, torch_dtype=torch.float16)

        print(f"Moving Bark model to device: {device}...")
        bark_model = bark_model.to(device)
        if str(device) != "cpu": xm.mark_step()
        print("Bark model successfully loaded and moved to device.")

        print(f"Tokenizing audio prompt: '{SAMPLE_PROMPT}'...")
        # Note: Bark can use voice_presets like "v2/en_speaker_6", etc.
        # For PoC, using default (None) which often picks a random English voice.
        inputs = bark_processor(SAMPLE_PROMPT, return_tensors="pt", voice_preset=None)
        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
        if str(device) != "cpu": xm.mark_step() # After moving inputs

        print("Generating audio waveform...")
        # Estimate max_length. Bark's token to time is variable. This is a heuristic.
        # Average tokens per sec for Bark is roughly 75 (24000 sample_rate / 320 hop_size for Encodec in Bark's setup)
        # So, for VIDEO_DURATION_S, it's VIDEO_DURATION_S * 75 tokens. Multiply by a factor for safety.
        # However, Bark's generate `max_length` is more about coarse tokens. Let's use a simpler heuristic for now.
        # A simpler heuristic often used for Bark: max_new_tokens. Default is often okay for short prompts.
        # For longer audio tied to video_duration_s, we can use a rough estimation based on expected tokens per second.
        # Bark's generation_config.max_length or similar might also provide hints.
        # Let's try a more direct estimation based on desired audio length and sampling rate for generate's max_length.
        # This parameter in `generate` is more like a safety cap on total tokens.
        estimated_max_tokens = int(AUDIO_SAMPLING_RATE * VIDEO_DURATION_S * BARK_MAX_LENGTH_HEURISTIC_MULTIPLIER / 100) # Very rough, adjust
        if BARK_MODEL_ID == "suno/bark-small": # Small model might need more tokens for same duration
             estimated_max_tokens = int(estimated_max_tokens * 1.2)


        with torch.no_grad():
            # Using generate method from BarkModel. `do_sample=True` is good for naturalness.
            speech_values = bark_model.generate(
                **inputs_on_device,
                do_sample=True, fine_temperature=0.7, coarse_temperature=0.7,
                # The 'max_length' or 'max_new_tokens' here is for the semantic token generation part of Bark.
                # It's not directly seconds. Let's use a sensible default or a small value for PoC.
                # Bark usually stops on its own with EOS token.
                # Forcing a very large max_length can sometimes lead to overly long silences or oddities.
                # The actual duration is more influenced by the input text length and content.
                # Let's use a moderate number of tokens that should be enough for a few seconds of speech.
                max_new_tokens=min(768, int(75 * VIDEO_DURATION_S * 1.5)) # Approx 75 tokens/sec, with a cap
            ).cpu().numpy().squeeze()
        if str(device) != "cpu": xm.mark_step()
        print("Audio waveform generated.")

        print(f"Saving audio to {AUDIO_FILE_PATH} at {AUDIO_SAMPLING_RATE} Hz...")
        scipy.io.wavfile.write(AUDIO_FILE_PATH, rate=AUDIO_SAMPLING_RATE, data=speech_values)
        print(f"Audio successfully saved to {AUDIO_FILE_PATH}")

    except Exception as e:
        print(f"An error occurred during audio generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bark_model is not None:
            del bark_model
            print("Bark model unloaded.")
        if bark_processor is not None:
            del bark_processor
            print("Bark processor unloaded.")
        gc.collect()
        if str(device) != "cpu": xm.mark_step()

    print("\nVideo and Audio generation (separate files) finished.")
    print(f"Silent Video: {SILENT_VIDEO_FILE_PATH}")
    print(f"Audio: {AUDIO_FILE_PATH}")

if __name__ == "__main__":
    print("Script for silent T2V and separate Bark audio generation (poc_t2v_with_audio.py).")
    print("To run: Ensure TPU setup, install deps, then call generate_video_and_audio_separately()")
    # generate_video_and_audio_separately() # Uncomment to run
    pass
