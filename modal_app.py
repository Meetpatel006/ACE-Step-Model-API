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
        "flask_cors",
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
        "azure-storage-blob",  # Azure Blob Storage support
        "uuid",  # For generating unique IDs
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

# Azure Blob Storage configuration
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "ace-step-audio")

def get_azure_blob_client():
    """Initialize Azure Blob Storage client"""
    from azure.storage.blob import BlobServiceClient
    
    # Get credentials from environment (injected by Modal secrets)
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = os.getenv("AZURE_STORAGE_KEY")
    blob_endpoint = os.getenv("AZURE_BLOB_ENDPOINT")
    
    # Debugging: Print the environment variables
    print(f"DEBUG: AZURE_STORAGE_ACCOUNT_NAME = {account_name}")
    print(f"DEBUG: AZURE_STORAGE_KEY = {'SET' if account_key else 'NOT SET'}")
    print(f"DEBUG: AZURE_BLOB_ENDPOINT = {blob_endpoint}")
    
    if not account_name or not account_key or not blob_endpoint:
        raise ValueError("Azure Storage credentials are missing. Please set AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_KEY, and AZURE_BLOB_ENDPOINT environment variables.")
    
    blob_service_client = BlobServiceClient(
        account_url=blob_endpoint,
        credential=account_key
    )
    return blob_service_client

def upload_to_azure_blob(file_path: str, blob_name: str) -> str:
    from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
    from datetime import datetime, timedelta
    try:
        # Absolute path
        file_path = str(Path(file_path).resolve())
        print(f"[Azure Upload] Preparing to upload {file_path} as {blob_name}")

        # Credentials
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        account_key = os.getenv("AZURE_STORAGE_KEY")
        blob_endpoint = os.getenv("AZURE_BLOB_ENDPOINT")
        container_name = os.getenv("AZURE_CONTAINER_NAME", "ace-step-audio")

        if not account_name or not account_key:
            raise ValueError("Azure Storage account name/key not set in environment variables.")

        # Create blob client
        blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net/",
            credential=account_key
        )
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Upload file
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True, max_concurrency=4)

        print(f"[Azure Upload] Upload complete for {blob_name}")

        # Generate SAS URL valid for 24 hours
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=24)
        )

        blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        print(f"[Azure Upload] Blob URL: {blob_url}")

        return blob_url

    except Exception as upload_error:
        print(f"[Azure Upload] Failed: {upload_error}", exc_info=True)
        return None


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
    from flask_cors import CORS
    from acestep.pipeline_ace_step import ACEStepPipeline
    import base64
    import gc
    import shutil
    import torch
    from werkzeug.middleware.proxy_fix import ProxyFix
    
    flask_app = Flask(__name__)
    CORS(flask_app, resources={r"/*": {"origins": "*"}})
    flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app)
    
    def env_flag(name: str, default: str = "0") -> bool:
        return os.environ.get(name, default).lower() in ("1", "true", "yes")
    
    device_id = int(os.environ.get("DEVICE_ID", 0))
    torch_compile = env_flag("TORCH_COMPILE", default="1")
    overlapped_decode = env_flag("OVERLAPPED_DECODE", default="1")
    
    @flask_app.route('/generate', methods=['POST'])
    def generate_song():
        import uuid
        data = request.get_json(force=True)
        gen_params = _extract_generation_params(data)

        try:
            # Load pipeline on demand
            pipeline = ACEStepPipeline(
                device_id=device_id,
                torch_compile=torch_compile,
                overlapped_decode=overlapped_decode,
                model_cache_dir="/models"
            )

            output_paths = pipeline(**gen_params)

        except Exception as e:
            return jsonify({'error': f'Pipeline execution failed: {e}'}), 500

        finally:
            # Release resources
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not output_paths:
            return jsonify({'error': 'Generation failed, no output paths'}), 500

        audio_path = output_paths[0]
        audio_id = str(uuid.uuid4())
        blob_name = f"ace-step-outputs/{audio_id}.wav"

        # Try uploading to Azure Blob
        try:
            audio_url = upload_to_azure_blob(audio_path, blob_name)

            # Remove local file
            try:
                os.remove(audio_path)
            except Exception as e:
                print(f"Failed to remove file {audio_path}: {e}")

            # Remove output directory
            try:
                shutil.rmtree(os.path.dirname(audio_path))
            except Exception as e:
                print(f"Failed to remove directory {os.path.dirname(audio_path)}: {e}")

            return jsonify({
                "audio_url": audio_url,
                "blob_name": blob_name,
                "audio_base64": None
            })

        except Exception as upload_error:
            print(f"Azure upload failed: {upload_error}")

            # Fallback: return base64
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            except Exception as read_error:
                return jsonify({'error': f'Failed to read audio file: {read_error}'}), 500

            # Cleanup outputs
            try:
                shutil.rmtree(os.path.dirname(audio_path))
            except Exception as e:
                print(f"Failed to clean directory {os.path.dirname(audio_path)}: {e}")

            return jsonify({"audio_base64": audio_b64})

            
        except Exception as e:
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500
    
    @flask_app.route('/generate/retake', methods=['POST'])
    def generate_retake():
        data = request.get_json(force=True)
        gen_params = _extract_generation_params(data)
        gen_params["task"] = "retake"
        if "retake_variance" not in data:
            gen_params["retake_variance"] = 0.2
        try:
            pipeline = ACEStepPipeline(
                device_id=device_id,
                torch_compile=torch_compile,
                overlapped_decode=overlapped_decode,
                model_cache_dir="/models",
            )
            output_paths = pipeline(**gen_params)
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not output_paths:
                return jsonify({'error': 'generation failed'}), 500
            audio_path = output_paths[0]
            
            # Generate a unique filename
            import uuid
            audio_id = str(uuid.uuid4())
            blob_name = f"ace-step-outputs/{audio_id}.wav"
            
            # Upload to Azure Blob Storage
            try:
                audio_url = upload_to_azure_blob(audio_path, blob_name)
                
                # Clean up local file
                try:
                    os.remove(audio_path)
                except Exception as e:
                    print(f"Failed to clean up local file {audio_path}: {e}")
                
                # Clean up directory
                try:
                    shutil.rmtree(os.path.dirname(audio_path))
                except Exception:
                    pass
                
                return jsonify({
                    "audio_url": audio_url,
                    "blob_name": blob_name,
                    "audio_base64": None
                })
            except Exception as upload_error:
                # If upload fails, fall back to base64
                print(f"Failed to upload to Azure Blob Storage: {upload_error}")
                
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                
                try:
                    shutil.rmtree(os.path.dirname(audio_path))
                except Exception:
                    pass
                
                return jsonify({"audio_base64": audio_b64})
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500

    @flask_app.route('/generate/repaint', methods=['POST'])
    def generate_repaint():
        data = request.get_json(force=True)
        gen_params = _extract_generation_params(data)
        gen_params["task"] = "repaint"
        try:
            pipeline = ACEStepPipeline(
                device_id=device_id,
                torch_compile=torch_compile,
                overlapped_decode=overlapped_decode,
                model_cache_dir="/models",
            )
            output_paths = pipeline(**gen_params)
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not output_paths:
                return jsonify({'error': 'generation failed'}), 500
            audio_path = output_paths[0]
            
            # Generate a unique filename
            import uuid
            audio_id = str(uuid.uuid4())
            blob_name = f"ace-step-outputs/{audio_id}.wav"
            
            # Upload to Azure Blob Storage
            try:
                audio_url = upload_to_azure_blob(audio_path, blob_name)
                
                # Clean up local file
                try:
                    os.remove(audio_path)
                except Exception as e:
                    print(f"Failed to clean up local file {audio_path}: {e}")
                
                # Clean up directory
                try:
                    shutil.rmtree(os.path.dirname(audio_path))
                except Exception:
                    pass
                
                return jsonify({
                    "audio_url": audio_url,
                    "blob_name": blob_name,
                    "audio_base64": None
                })
            except Exception as upload_error:
                # If upload fails, fall back to base64
                print(f"Failed to upload to Azure Blob Storage: {upload_error}")
                
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                
                try:
                    shutil.rmtree(os.path.dirname(audio_path))
                except Exception:
                    pass
                
                return jsonify({"audio_base64": audio_b64})
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500

    @flask_app.route('/generate/edit', methods=['POST'])
    def generate_edit():
        data = request.get_json(force=True)
        gen_params = _extract_generation_params(data)
        gen_params["task"] = "edit"
        if not gen_params.get("edit_target_prompt"):
            gen_params["edit_target_prompt"] = gen_params.get("prompt")
        if not gen_params.get("edit_target_lyrics"):
            gen_params["edit_target_lyrics"] = gen_params.get("lyrics")
        try:
            pipeline = ACEStepPipeline(
                device_id=device_id,
                torch_compile=torch_compile,
                overlapped_decode=overlapped_decode,
                model_cache_dir="/models",
            )
            output_paths = pipeline(**gen_params)
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not output_paths:
                return jsonify({'error': 'generation failed'}), 500
            audio_path = output_paths[0]
            
            # Generate a unique filename
            import uuid
            audio_id = str(uuid.uuid4())
            blob_name = f"ace-step-outputs/{audio_id}.wav"
            
            # Upload to Azure Blob Storage
            try:
                audio_url = upload_to_azure_blob(audio_path, blob_name)
                
                # Clean up local file
                try:
                    os.remove(audio_path)
                except Exception as e:
                    print(f"Failed to clean up local file {audio_path}: {e}")
                
                # Clean up directory
                try:
                    shutil.rmtree(os.path.dirname(audio_path))
                except Exception:
                    pass
                
                return jsonify({
                    "audio_url": audio_url,
                    "blob_name": blob_name,
                    "audio_base64": None
                })
            except Exception as upload_error:
                # If upload fails, fall back to base64
                print(f"Failed to upload to Azure Blob Storage: {upload_error}")
                
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                
                try:
                    shutil.rmtree(os.path.dirname(audio_path))
                except Exception:
                    pass
                
                return jsonify({"audio_base64": audio_b64})
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500

    @flask_app.route('/generate/extend', methods=['POST'])
    def generate_extend():
        data = request.get_json(force=True)
        gen_params = _extract_generation_params(data)
        gen_params["task"] = "extend"

        left_extend_length = data.get("left_extend_length", None)
        right_extend_length = data.get("right_extend_length", None)
        if left_extend_length is not None or right_extend_length is not None:
            try:
                left_len = float(left_extend_length or 0.0)
                right_len = float(right_extend_length or 0.0)
                repaint_start = -left_len
                repaint_end = float(gen_params.get("audio_duration", 0.0)) + right_len
                gen_params["repaint_start"] = repaint_start
                gen_params["repaint_end"] = repaint_end
            except Exception:
                pass

        if "retake_variance" not in data:
            gen_params["retake_variance"] = 1.0
        if "extend_seeds" in data and not gen_params.get("retake_seeds"):
            gen_params["retake_seeds"] = data.get("extend_seeds")
        try:
            pipeline = ACEStepPipeline(
                device_id=device_id,
                torch_compile=torch_compile,
                overlapped_decode=overlapped_decode,
                model_cache_dir="/models",
            )
            output_paths = pipeline(**gen_params)
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not output_paths:
                return jsonify({'error': 'generation failed'}), 500
            audio_path = output_paths[0]
            
            # Generate a unique filename
            import uuid
            audio_id = str(uuid.uuid4())
            blob_name = f"ace-step-outputs/{audio_id}.wav"
            
            # Upload to Azure Blob Storage
            try:
                audio_url = upload_to_azure_blob(audio_path, blob_name)
                
                # Clean up local file
                try:
                    os.remove(audio_path)
                except Exception as e:
                    print(f"Failed to clean up local file {audio_path}: {e}")
                
                # Clean up directory
                try:
                    shutil.rmtree(os.path.dirname(audio_path))
                except Exception:
                    pass
                
                return jsonify({
                    "audio_url": audio_url,
                    "blob_name": blob_name,
                    "audio_base64": None
                })
            except Exception as upload_error:
                # If upload fails, fall back to base64
                print(f"Failed to upload to Azure Blob Storage: {upload_error}")
                
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                
                try:
                    shutil.rmtree(os.path.dirname(audio_path))
                except Exception:
                    pass
                
                return jsonify({"audio_base64": audio_b64})
        except Exception as e:
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
    Verify ACE Step package installation and setup Azure Blob Storage
    """
    import sys
    
    print("Verifying ACE Step installation...")
    
    # Verify installation
    try:
        from acestep.pipeline_ace_step import ACEStepPipeline
        print("ACE Step pipeline import successful")
        
        # Try to set up Azure Blob Storage
        try:
            blob_client = get_azure_blob_client()
            # Try to create container if it doesn't exist
            try:
                container_client = blob_client.get_container_client(AZURE_CONTAINER_NAME)
                container_client.get_container_properties()
                print(f"Container '{AZURE_CONTAINER_NAME}' already exists")
            except Exception:
                try:
                    blob_client.create_container(AZURE_CONTAINER_NAME)
                    print(f"Created container '{AZURE_CONTAINER_NAME}'")
                except Exception as e:
                    print(f"Failed to create container '{AZURE_CONTAINER_NAME}': {e}")
        except Exception as e:
            print(f"Warning: Azure Blob Storage not configured: {e}")
            print("Continuing without Azure Blob Storage support...")
        
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