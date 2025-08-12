import modal
import os
import subprocess
import sys
from pathlib import Path

# Define the Modal app
app = modal.App("ace-step-flask-api")

# Define the image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "git",
        "wget",
        "curl",
        "ffmpeg",
        "libsndfile1",
        "libsndfile1-dev",
        "build-essential",  # Often needed for compilation
    ])
    .pip_install([
        "flask",
        "requests",
        "torch",
        "torchaudio",
        "transformers",
        "accelerate",
        "diffusers",
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "pydub",
        "werkzeug",
    ])
    .run_commands([
        # Create directory for ace-step
        "mkdir -p /app",
        # Clone and install ACE-Step from GitHub
        "cd /app && git clone https://github.com/ace-step/ACE-Step.git",
        "cd /app/ACE-Step && pip install -e .",
    ])
)

# Create a volume for model storage and temporary files
model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
temp_volume = modal.Volume.from_name("ace-step-temp", create_if_missing=True)


# Helper to extract generation parameters from request dict with safe defaults.
def _extract_generation_params(request_dict: dict) -> dict:
    """
    Normalize and collect all supported generation parameters from the request body.
    Keeps backward compatibility with minimal payloads (prompt/lyrics/length).
    """
    data = request_dict or {}

    # Back-compat: allow either audio_duration or length
    audio_duration = data.get("audio_duration", data.get("length", 60))
    try:
        audio_duration = float(audio_duration)
    except Exception:
        audio_duration = 60.0

    def _float(name: str, default: float) -> float:
        value = data.get(name, default)
        try:
            return float(value)
        except Exception:
            return default

    def _int(name: str, default: int) -> int:
        value = data.get(name, default)
        try:
            return int(value)
        except Exception:
            return default

    def _bool(name: str, default: bool) -> bool:
        value = data.get(name, default)
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    params = {
        # core
        "format": data.get("format", "wav"),
        "audio_duration": audio_duration,
        "prompt": data.get("prompt", ""),
        "lyrics": data.get("lyrics", ""),

        # basic settings
        "infer_step": _int("infer_step", 60),
        "guidance_scale": _float("guidance_scale", 15.0),
        "scheduler_type": data.get("scheduler_type", "euler"),
        "cfg_type": data.get("cfg_type", "apg"),
        "omega_scale": _float("omega_scale", 10.0),
        "manual_seeds": data.get("manual_seeds", None),
        "guidance_interval": _float("guidance_interval", 0.5),
        "guidance_interval_decay": _float("guidance_interval_decay", 0.0),
        "min_guidance_scale": _float("min_guidance_scale", 3.0),
        "use_erg_tag": _bool("use_erg_tag", True),
        "use_erg_lyric": _bool("use_erg_lyric", True),
        "use_erg_diffusion": _bool("use_erg_diffusion", True),
        "oss_steps": data.get("oss_steps", None),  # accepts list or comma-separated string
        "guidance_scale_text": _float("guidance_scale_text", 0.0),
        "guidance_scale_lyric": _float("guidance_scale_lyric", 0.0),

        # audio2audio
        "audio2audio_enable": _bool("audio2audio_enable", False),
        "ref_audio_strength": _float("ref_audio_strength", 0.5),
        "ref_audio_input": data.get("ref_audio_input", None),

        # LoRA
        "lora_name_or_path": data.get("lora_name_or_path", "none"),
        "lora_weight": _float("lora_weight", 1.0),

        # retake/repaint/edit/extend controls
        "retake_seeds": data.get("retake_seeds", None),
        "retake_variance": _float("retake_variance", 0.5),
        "task": data.get("task", "text2music"),
        "repaint_start": _float("repaint_start", 0.0),
        "repaint_end": _float("repaint_end", 0.0),
        "src_audio_path": data.get("src_audio_path", None),
        "edit_target_prompt": data.get("edit_target_prompt", None),
        "edit_target_lyrics": data.get("edit_target_lyrics", None),
        "edit_n_min": _float("edit_n_min", 0.0),
        "edit_n_max": _float("edit_n_max", 1.0),

        # misc
        "batch_size": _int("batch_size", 1),
        "save_path": data.get("save_path", None),
        "debug": _bool("debug", False),
    }

    return params

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/models": model_volume,
        "/tmp/ace_outputs": temp_volume,
    },
    timeout=600,  # 10 minute timeout for generation
)
@modal.fastapi_endpoint(
    method="POST",
    label="ace-step-generate",
)
def generate_song(request_data: dict):
    """
    Modal web endpoint that wraps the Flask functionality
    """
    from acestep.pipeline_ace_step import ACEStepPipeline
    import base64
    import gc
    import shutil
    import torch
    import tempfile
    import json
    
    # Gather generation parameters (backward compatible)
    gen_params = _extract_generation_params(request_data)
    
    # Environment configuration
    device_id = 0  # Use first GPU
    torch_compile = True
    overlapped_decode = True
    
    try:
        # Load the pipeline on demand
        pipeline = ACEStepPipeline(
            device_id=device_id,
            torch_compile=torch_compile,
            overlapped_decode=overlapped_decode,
            model_cache_dir="/models",  # Use the mounted volume
        )
        
        # Call the ACE Step pipeline with full parameter set
        output_paths = pipeline(**gen_params)
        
        # Explicitly release memory used by the pipeline
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if not output_paths:
            return {"error": "generation failed"}, 500
        
        audio_path = output_paths[0]
        
        # Read and encode audio file
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Clean up output files
        outputs_dir = os.path.dirname(audio_path)
        try:
            shutil.rmtree(outputs_dir)
        except Exception as e:
            print(f"Warning: Failed to clean outputs directory {outputs_dir}: {e}")
        
        return {"audio_base64": audio_b64}
        
    except Exception as e:
        # Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {"error": f"Generation failed: {str(e)}"}, 500


# Alternative approach: Run the Flask app directly in Modal
@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/models": model_volume,
        "/tmp/ace_outputs": temp_volume,
    },
    scaledown_window=300,
)
@modal.wsgi_app()
def flask_app():
    """
    Run the Flask app directly in Modal using WSGI
    """
    from flask import Flask, request, jsonify
    from acestep.pipeline_ace_step import ACEStepPipeline
    import base64
    import gc
    import shutil
    import torch
    from werkzeug.middleware.proxy_fix import ProxyFix
    
    flask_app = Flask(__name__)
    flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app)
    
    def env_flag(name: str, default: str = "0") -> bool:
        return os.environ.get(name, default).lower() in ("1", "true", "yes")
    
    device_id = int(os.environ.get("DEVICE_ID", 0))
    torch_compile = env_flag("TORCH_COMPILE", default="1")
    overlapped_decode = env_flag("OVERLAPPED_DECODE", default="1")
    
    @flask_app.route('/generate', methods=['POST'])
    def generate_song():
        data = request.get_json(force=True)
        gen_params = _extract_generation_params(data)
        
        try:
            # Load the pipeline on demand
            pipeline = ACEStepPipeline(
                device_id=device_id,
                torch_compile=torch_compile,
                overlapped_decode=overlapped_decode,
                model_cache_dir="/models",  # Use mounted volume
            )
            
            # Call the ACE Step pipeline
            output_paths = pipeline(**gen_params)
            
            # Explicitly release memory used by the pipeline
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if not output_paths:
                return jsonify({'error': 'generation failed'}), 500
            
            audio_path = output_paths[0]
            
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            # Clean up
            outputs_dir = os.path.dirname(audio_path)
            try:
                shutil.rmtree(outputs_dir)
            except Exception as e:
                flask_app.logger.warning(f"Failed to clean outputs directory {outputs_dir}: {e}")
            
            return jsonify({"audio_base64": audio_b64})
            
        except Exception as e:
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500
    
    @flask_app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy"})
    
    return flask_app


# Function to setup ACE Step package from local mount
@app.function(
    image=image,
    volumes={"/models": model_volume},
    timeout=1800,  # 30 minutes for setup
)
def setup_ace_step():
    """
    Verify ACE Step package installation
    """
    import sys
    
    print("Verifying ACE Step installation...")
    
    # Verify installation
    try:
        from acestep.pipeline_ace_step import ACEStepPipeline
        print("ACE Step pipeline import successful")
        return True
    except ImportError as e:
        print(f"Failed to import ACE Step pipeline: {e}")
        print("Available packages:")
        import pkg_resources
        for pkg in pkg_resources.working_set:
            if 'ace' in pkg.project_name.lower():
                print(f"  {pkg.project_name}: {pkg.version}")
        raise


# Function to copy additional files if needed
@app.function(
    image=image,
    volumes={
        "/models": model_volume,
        "/tmp/ace_outputs": temp_volume,
    },
)
def setup_directories():
    """
    Setup necessary directories
    """
    from pathlib import Path
    
    # Create necessary directories
    Path("/models").mkdir(exist_ok=True)
    Path("/tmp/ace_outputs").mkdir(exist_ok=True)
    
    print("Directory setup complete!")


# Local development function
@app.local_entrypoint()
def main():
    """
    Entry point for running setup and deployment
    """
    print("Verifying ACE Step installation...")
    setup_ace_step.remote()
    
    print("Setting up directories...")
    setup_directories.remote()
    
    print("Setup complete! Your Flask API is now deployed on Modal.")
    print("You can access it at the generated URL from Modal dashboard.")


# If you want to run this locally for testing
if __name__ == "__main__":
    # For local testing, you can run the setup
    import modal
    with modal.App("ace-step-flask-api").run():
        setup_ace_step.remote()
        setup_directories.remote()
        print("Setup complete!")