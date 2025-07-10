# Step 2: Implement Basic Text-to-Video (Silent)
# This script should be run in a Colab cell AFTER
# Step 1 (Setup Colab Environment & Install Dependencies) is complete and verified.

import torch
import torch_xla.core.xla_model as xm
import diffusers
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler # DPMSolver often works well
from diffusers.utils import export_to_video
import os
import gc

# --- Configuration ---
T2V_MODEL_ID = "cerspense/zeroscope_v2_576w"
# Alternative if _576w is too small for quality test later: "cerspense/zeroscope_v2_XL" (larger, needs more VRAM)

SAMPLE_PROMPT = "A cute puppy playing in a field of flowers"
NEGATIVE_PROMPT = "ugly, blurry, low quality, watermark, text, deformed"
SILENT_VIDEO_FILE_PATH = "generated_silent_video.mp4"

# PoC parameters (lowered for faster initial testing and less memory usage)
VIDEO_DURATION_S = 2  # seconds
VIDEO_FPS = 12        # frames per second
NUM_FRAMES = int(VIDEO_DURATION_S * VIDEO_FPS)
NUM_INFERENCE_STEPS = 20 # Number of denoising steps
GUIDANCE_SCALE = 9.0   # How much to adhere to the prompt

# --- Helper Functions (from previous PoC, adapted) ---
def get_xla_device_or_cpu():
    """Acquires the XLA device (TPU). Falls back to CPU if TPU is not available."""
    try:
        device = xm.xla_device()
        print(f"Successfully acquired XLA device: {device}")
        return device
    except Exception as e:
        print(f"XLA device not available: {e}. Falling back to CPU (expect very slow performance).")
        return torch.device("cpu")

def cleanup_file_if_exists(filename):
    """Removes a specified file if it exists."""
    if os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"Successfully removed existing file: {filename}")
        except Exception as e:
            print(f"Error removing file {filename}: {e}")

# --- Main T2V Logic ---
def generate_silent_t2v():
    """
    Generates a silent video from a text prompt using ZeroScope model on TPU.
    """
    print(f"Starting silent Text-to-Video generation with model: {T2V_MODEL_ID}")
    device = get_xla_device_or_cpu()
    cleanup_file_if_exists(SILENT_VIDEO_FILE_PATH)

    t2v_pipe = None
    try:
        # 1. Load the Text-to-Video Pipeline
        # ZeroScope models are typically used with DiffusionPipeline,
        # which can infer the correct sub-pipeline (like TextToVideoSDPipeline).
        print(f"Loading T2V pipeline for {T2V_MODEL_ID}...")
        t2v_pipe = DiffusionPipeline.from_pretrained(
            T2V_MODEL_ID,
            torch_dtype=torch.float16 # Use float16 for memory efficiency on TPU
        )

        # Use a suitable scheduler
        t2v_pipe.scheduler = DPMSolverMultistepScheduler.from_config(t2v_pipe.scheduler.config)

        # Move pipeline to the XLA device
        # This is a critical step. If the model is large, this might take time or cause OOM.
        print(f"Moving T2V pipeline to device: {device}...")
        t2v_pipe = t2v_pipe.to(device)
        if str(device) != "cpu": xm.mark_step() # Mark step after model transfer
        print("T2V pipeline successfully loaded and moved to device.")

        # (Optional) Enable model CPU offload if memory is an issue,
        # though its effectiveness and performance with XLA needs careful testing.
        # May not be needed for zeroscope_v2_576w if TPU has enough memory.
        # if hasattr(t2v_pipe, 'enable_model_cpu_offload'):
        #     print("Enabling model CPU offload (if applicable)...")
        #     t2v_pipe.enable_model_cpu_offload()

        # 2. Generate Video Frames
        print(f"Generating video frames for prompt: '{SAMPLE_PROMPT}'...")
        print(f"Parameters: {NUM_FRAMES} frames, {NUM_INFERENCE_STEPS} steps, guidance {GUIDANCE_SCALE}")

        # Generator for reproducibility if desired (optional for PoC)
        # generator = torch.Generator(device=device).manual_seed(42)

        video_frames_tensor = t2v_pipe(
            prompt=SAMPLE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=320, # zeroscope_v2_576w is often trained at 320x576 or similar
            width=576,  # ensure dimensions are multiples of model's VAE downscale factor (usually 8 or 16)
            # generator=generator # Pass generator if using one
        ).frames # .frames should give a list of PIL Images or a tensor

        if str(device) != "cpu": xm.mark_step() # Mark step after inference
        print("Video frames generated successfully.")

        # 3. Export to Video File
        # The 'frames' output from ZeroScope pipelines in diffusers is typically a list of PIL Images or a tensor.
        # export_to_video expects a list of PIL Images or a NumPy array (NumFrames, Height, Width, Channels)
        # or a torch tensor (NumFrames, Channels, Height, Width)
        print(f"Exporting frames to video: {SILENT_VIDEO_FILE_PATH} at {VIDEO_FPS} FPS...")
        export_to_video(video_frames_tensor, SILENT_VIDEO_FILE_PATH, fps=VIDEO_FPS)
        print(f"Silent video successfully saved to {SILENT_VIDEO_FILE_PATH}")

        # In Colab, display the video:
        print(f"\nTo display video in Colab, use: from IPython.display import Video; Video('{SILENT_VIDEO_FILE_PATH}', embed=True)")

    except Exception as e:
        print(f"An error occurred during silent T2V generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup to free VRAM/RAM
        if t2v_pipe is not None:
            del t2v_pipe
            print("T2V pipeline unloaded.")
        gc.collect()
        if str(device) != "cpu":
            # Potentially more XLA cleanup if needed, but xm.mark_step() is key during operations
            pass
        print("Silent T2V generation process finished.")

if __name__ == "__main__":
    # This block is for direct script execution.
    # In Colab, you would call generate_silent_t2v() in a cell.
    print("Script for silent Text-to-Video generation (poc_silent_t2v.py).")
    print("To run the PoC in Colab: Ensure TPU is set up, dependencies are installed, then call generate_silent_t2v()")
    # generate_silent_t2v() # Uncomment to run if executing script directly (ensure environment is set up)
    pass
