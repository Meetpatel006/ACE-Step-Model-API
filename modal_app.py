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
    
    # Get request data
    prompt = request_data.get('prompt', '')
    lyrics = request_data.get('lyrics', '')
    length = float(request_data.get('length', 60))
    
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
        
        # Call the ACE Step pipeline
        output_paths = pipeline(
            audio_duration=length,
            prompt=prompt,
            lyrics=lyrics,
        )
        
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
        prompt = data.get('prompt', '')
        lyrics = data.get('lyrics', '')
        length = float(data.get('length', 60))
        
        try:
            # Load the pipeline on demand
            pipeline = ACEStepPipeline(
                device_id=device_id,
                torch_compile=torch_compile,
                overlapped_decode=overlapped_decode,
                model_cache_dir="/models",  # Use mounted volume
            )
            
            # Call the ACE Step pipeline
            output_paths = pipeline(
                audio_duration=length,
                prompt=prompt,
                lyrics=lyrics,
            )
            
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