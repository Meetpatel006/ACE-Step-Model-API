import argparse
import base64
import json
import os
import time
from pathlib import Path
import requests
from typing import Any, Dict, Optional, List


# Predefined parameter presets. These get merged into the payload when --preset is used.
PRESETS: Dict[str, Dict[str, Any]] = {
    # Balanced defaults
    "default": {
        "format": "wav",
        "infer_step": 60,
        "guidance_scale": 15.0,
        "scheduler_type": "euler",
        "cfg_type": "apg",
        "omega_scale": 10.0,
        "guidance_interval": 0.5,
        "guidance_interval_decay": 0.0,
        "min_guidance_scale": 3.0,
        "use_erg_tag": True,
        "use_erg_lyric": False,
        "use_erg_diffusion": True,
    },
    # Faster generations, slightly lower quality
    "fast": {
        "format": "wav",
        "infer_step": 30,
        "guidance_scale": 10.0,
        "scheduler_type": "euler",
        "cfg_type": "apg",
        "omega_scale": 8.0,
    },
    # Higher quality (slower)
    "quality": {
        "format": "wav",
        "infer_step": 90,
        "guidance_scale": 17.0,
        "scheduler_type": "euler",
        "cfg_type": "apg",
        "omega_scale": 12.0,
    },
    # Heun scheduler + classic CFG
    "heun_cfg": {
        "scheduler_type": "heun",
        "cfg_type": "cfg",
        "guidance_scale": 12.0,
        "infer_step": 40,
    },
    # PingPong SDE with guidance window/decay
    "pingpong_window": {
        "scheduler_type": "pingpong",
        "guidance_interval": 0.3,
        "guidance_interval_decay": 0.5,
        "min_guidance_scale": 2.0,
        "infer_step": 30,
    },
    # Disable ERG everywhere
    "erg_off": {
        "use_erg_tag": False,
        "use_erg_lyric": False,
        "use_erg_diffusion": False,
    },
    # Double-condition guidance (text + lyric)
    "double_condition_guidance": {
        "guidance_scale_text": 5.0,
        "guidance_scale_lyric": 1.5,
        "guidance_scale": 7.5,
    },
    # Audio-to-audio template (requires --ref-audio-input)
    "audio2audio": {
        "audio2audio_enable": True,
        "ref_audio_strength": 0.5,
    },
}

# Full default values for every supported parameter. Used when sending
# a complete payload (e.g., when --preset is provided or --send-all-params).
ALL_DEFAULTS: Dict[str, Any] = {
    # core
    "format": "wav",
    "audio_duration": None,  # will prefer args.audio_duration or args.length
    "prompt": "",
    "lyrics": "",

    # basic settings
    "infer_step": 60,
    "guidance_scale": 15.0,
    "scheduler_type": "euler",
    "cfg_type": "apg",
    "omega_scale": 10.0,
    "manual_seeds": None,
    "guidance_interval": 0.5,
    "guidance_interval_decay": 0.0,
    "min_guidance_scale": 3.0,
    "use_erg_tag": True,
    "use_erg_lyric": False,
    "use_erg_diffusion": True,
    "oss_steps": None,
    "guidance_scale_text": 0.0,
    "guidance_scale_lyric": 0.0,

    # audio2audio
    "audio2audio_enable": False,
    "ref_audio_strength": 0.5,
    "ref_audio_input": None,

    # LoRA
    "lora_name_or_path": "none",
    "lora_weight": 1.0,

    # retake/repaint/edit/extend
    "task": "text2music",
    "retake_seeds": None,
    "retake_variance": 0.5,
    "repaint_start": 0.0,
    "repaint_end": 0.0,
    "src_audio_path": None,
    "edit_target_prompt": None,
    "edit_target_lyrics": None,
    "edit_n_min": 0.0,
    "edit_n_max": 1.0,

    # misc
    "batch_size": 1,
    "save_path": None,
    "debug": False,
}


def _str_to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    return None


def _maybe_add(payload: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def _assemble_payload_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a request payload from CLI args, preserving backward compatibility."""
    payload: Dict[str, Any] = {
        "prompt": args.prompt,
        "lyrics": args.lyrics,
        # Backward compatibility: keep sending 'length' unless explicit audio_duration is provided
        "length": args.length,
    }

    # Merge preset first (so explicit CLI flags can override below)
    if getattr(args, "preset", None):
        preset = PRESETS.get(args.preset)
        if preset:
            payload.update(preset)

    # Prefer explicit audio_duration over length if provided
    if getattr(args, "audio_duration", None) is not None:
        payload.pop("length", None)
        payload["audio_duration"] = args.audio_duration

    # Optional fields (only add when provided)
    _maybe_add(payload, "format", args.format)
    _maybe_add(payload, "infer_step", args.infer_step)
    _maybe_add(payload, "guidance_scale", args.guidance_scale)
    _maybe_add(payload, "scheduler_type", args.scheduler_type)
    _maybe_add(payload, "cfg_type", args.cfg_type)
    _maybe_add(payload, "omega_scale", args.omega_scale)
    _maybe_add(payload, "manual_seeds", args.manual_seeds)
    _maybe_add(payload, "guidance_interval", args.guidance_interval)
    _maybe_add(payload, "guidance_interval_decay", args.guidance_interval_decay)
    _maybe_add(payload, "min_guidance_scale", args.min_guidance_scale)

    use_erg_tag_bool = _str_to_bool(args.use_erg_tag)
    use_erg_lyric_bool = _str_to_bool(args.use_erg_lyric)
    use_erg_diffusion_bool = _str_to_bool(args.use_erg_diffusion)
    _maybe_add(payload, "use_erg_tag", use_erg_tag_bool)
    _maybe_add(payload, "use_erg_lyric", use_erg_lyric_bool)
    _maybe_add(payload, "use_erg_diffusion", use_erg_diffusion_bool)

    _maybe_add(payload, "oss_steps", args.oss_steps)
    _maybe_add(payload, "guidance_scale_text", args.guidance_scale_text)
    _maybe_add(payload, "guidance_scale_lyric", args.guidance_scale_lyric)

    audio2audio_enable_bool = _str_to_bool(args.audio2audio_enable)
    debug_bool = _str_to_bool(args.debug)
    _maybe_add(payload, "audio2audio_enable", audio2audio_enable_bool)
    _maybe_add(payload, "ref_audio_strength", args.ref_audio_strength)
    _maybe_add(payload, "ref_audio_input", args.ref_audio_input)
    _maybe_add(payload, "lora_name_or_path", args.lora_name_or_path)
    _maybe_add(payload, "lora_weight", args.lora_weight)

    _maybe_add(payload, "task", args.task)
    _maybe_add(payload, "retake_seeds", args.retake_seeds)
    _maybe_add(payload, "retake_variance", args.retake_variance)
    _maybe_add(payload, "repaint_start", args.repaint_start)
    _maybe_add(payload, "repaint_end", args.repaint_end)
    _maybe_add(payload, "src_audio_path", args.src_audio_path)
    _maybe_add(payload, "edit_target_prompt", args.edit_target_prompt)
    _maybe_add(payload, "edit_target_lyrics", args.edit_target_lyrics)
    _maybe_add(payload, "edit_n_min", args.edit_n_min)
    _maybe_add(payload, "edit_n_max", args.edit_n_max)

    _maybe_add(payload, "batch_size", args.batch_size)
    _maybe_add(payload, "save_path", args.save_path)
    _maybe_add(payload, "debug", debug_bool)

    # If using a preset or explicitly requested, ensure all keys are present
    send_all = bool(getattr(args, "preset", None)) or bool(getattr(args, "send_all_params", False))
    if send_all:
        # Start from defaults, overlay current payload
        full_payload = dict(ALL_DEFAULTS)
        # Respect provided duration/length
        if "audio_duration" not in payload and "length" in payload:
            # We'll keep 'length' and won't set audio_duration so server back-compat stays
            pass
        full_payload.update(payload)

        # Audio2Audio mode selection
        mode = getattr(args, "send_all_params_mode", "plain")
        if mode == "audio2audio":
            full_payload["audio2audio_enable"] = True
            # keep ref_audio_strength/defaults; ref_audio_input can come from CLI
        else:
            full_payload["audio2audio_enable"] = False

        # If neither audio_duration nor length present, fall back to args.length
        if "audio_duration" not in full_payload and "length" not in full_payload:
            full_payload["length"] = getattr(args, "length", 5)

        return full_payload

    return payload


def _send_request(url: str, payload: Dict[str, Any], output_path: Path, dry_run: bool = False) -> Path:
    """Send one generation request and save audio to output_path. Returns the saved path."""
    if dry_run:
        print(json.dumps({"url": url, "payload": payload}, indent=2))
        return output_path

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise SystemExit(f"Request failed: {response.status_code} {response.text}")

    response_json = response.json()
    
    # Check if we have a blob URL (new Azure implementation)
    audio_url = response_json.get("audio_url")
    if audio_url:
        # Download audio from Azure Blob Storage
        audio_response = requests.get(audio_url)
        if audio_response.status_code != 200:
            raise SystemExit(f"Failed to download audio from blob storage: {audio_response.status_code}")
        
        audio_bytes = audio_response.content
        blob_name = response_json.get("blob_name", "unknown")
        print(f"Downloaded audio from Azure Blob Storage (blob: {blob_name})")
    else:
        # Fallback to base64 (old implementation or if Azure upload failed)
        audio_base64 = response_json.get("audio_base64")
        if not audio_base64:
            raise SystemExit("Response missing audio data")
        
        audio_bytes = base64.b64decode(audio_base64)
        print("Downloaded audio using base64 encoding")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    print(f"Saved song to {str(output_path)}")
    return output_path


def _run_auto_tests(args: argparse.Namespace) -> None:
    """Run a small suite of safe, text2music-only tests that exercise key parameters."""
    base_prompt = args.prompt or "funk, pop, energetic"
    base_lyrics = args.lyrics or "[instrumental]"
    base_duration = args.audio_duration if args.audio_duration is not None else args.length
    audio_format = args.format or "wav"

    tests: List[Dict[str, Any]] = [
        {"name": "default_euler_apg", "scheduler_type": "euler", "cfg_type": "apg", "infer_step": 40},
        {"name": "heun_cfg", "scheduler_type": "heun", "cfg_type": "cfg", "guidance_scale": 12.0, "infer_step": 35},
        {"name": "pingpong_guidance_window", "scheduler_type": "pingpong", "guidance_interval": 0.3, "guidance_interval_decay": 0.5, "min_guidance_scale": 2.0, "infer_step": 30},
        {"name": "erg_off", "use_erg_tag": "false", "use_erg_diffusion": "false", "use_erg_lyric": "false"},
        {"name": "granularity_high", "omega_scale": 20.0},
        {"name": "double_condition_guidance", "guidance_scale_text": 5.0, "guidance_scale_lyric": 1.5, "guidance_scale": 7.5},
    ]

    out_dir = Path(args.test_output_dir or "test_outputs")
    timestamp = time.strftime("%Y%m%d%H%M%S")

    for case in tests:
        # Build a fresh args-like object overlaying the case onto provided args
        payload = {
            "format": audio_format,
            "audio_duration": base_duration,
            "prompt": base_prompt,
            "lyrics": base_lyrics,
        }
        payload.update(case)

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Output filename
        filename = f"{timestamp}_{case['name']}.{audio_format}"
        output_path = out_dir / filename

        print(f"\n[auto-test] Running case: {case['name']}")
        _send_request(args.url, payload, output_path, dry_run=bool(args.dry_run))


def _run_manual_test(args: argparse.Namespace) -> None:
    """Prompt the user for parameters interactively and send one request."""
    print("Manual test mode: press Enter to keep defaults / skip.")
    base: Dict[str, Any] = {
        "prompt": input(f"prompt [{args.prompt}]: ") or args.prompt,
        "lyrics": input(f"lyrics [{args.lyrics}]: ") or args.lyrics,
    }

    # Duration
    dur_str = input(f"audio_duration (seconds) [{args.audio_duration or args.length}]: ")
    if dur_str.strip():
        try:
            base["audio_duration"] = float(dur_str)
        except ValueError:
            print("Invalid duration; skipping")
    else:
        # fallback to provided flags
        if args.audio_duration is not None:
            base["audio_duration"] = args.audio_duration
        else:
            base["length"] = args.length

    # Simple toggles/choices
    fmt = input(f"format [mp3|ogg|flac|wav] [{args.format or 'wav'}]: ") or (args.format or "wav")
    base["format"] = fmt
    sched = input(f"scheduler_type [euler|heun|pingpong] [{args.scheduler_type or 'euler'}]: ") or (args.scheduler_type or "euler")
    base["scheduler_type"] = sched
    cfg = input(f"cfg_type [cfg|apg|cfg_star] [{args.cfg_type or 'apg'}]: ") or (args.cfg_type or "apg")
    base["cfg_type"] = cfg

    # Numerics (optional)
    for key, cur in [
        ("infer_step", args.infer_step or 60),
        ("guidance_scale", args.guidance_scale or 15.0),
        ("omega_scale", args.omega_scale or 10.0),
        ("guidance_interval", args.guidance_interval or 0.5),
        ("guidance_interval_decay", args.guidance_interval_decay or 0.0),
        ("min_guidance_scale", args.min_guidance_scale or 3.0),
        ("guidance_scale_text", args.guidance_scale_text or 0.0),
        ("guidance_scale_lyric", args.guidance_scale_lyric or 0.0),
    ]:
        val = input(f"{key} [{cur}]: ")
        if val.strip():
            try:
                base[key] = float(val) if "." in val else int(val)
            except ValueError:
                print(f"Invalid {key}; skipping")

    # Strings
    manual_seeds = input(f"manual_seeds (e.g. 1,2,3) [{args.manual_seeds or ''}]: ") or args.manual_seeds
    if manual_seeds:
        base["manual_seeds"] = manual_seeds
    oss_steps = input(f"oss_steps (e.g. 16,29,52) [{args.oss_steps or ''}]: ") or args.oss_steps
    if oss_steps:
        base["oss_steps"] = oss_steps

    # Booleans
    for key, cur in [
        ("use_erg_tag", args.use_erg_tag),
        ("use_erg_lyric", args.use_erg_lyric),
        ("use_erg_diffusion", args.use_erg_diffusion),
        ("audio2audio_enable", args.audio2audio_enable),
        ("debug", args.debug),
    ]:
        val = input(f"{key} [true|false] [{cur or ''}]: ")
        b = _str_to_bool(val) if val.strip() else _str_to_bool(cur)
        if b is not None:
            base[key] = b

    # Misc
    lora = input(f"lora_name_or_path [{args.lora_name_or_path or 'none'}]: ") or (args.lora_name_or_path or "none")
    base["lora_name_or_path"] = lora
    lw = input(f"lora_weight [{args.lora_weight or 1.0}]: ")
    if lw.strip():
        try:
            base["lora_weight"] = float(lw)
        except ValueError:
            print("Invalid lora_weight; skipping")

    # Send request
    out_dir = Path(args.test_output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d%H%M%S")
    ext = base.get("format", "wav")
    output_file = out_dir / f"manual_{ts}.{ext}"
    _send_request(args.url, base, output_file, dry_run=bool(args.dry_run))


def main():
    parser = argparse.ArgumentParser(description="Call the ACE Step service")

    # Minimal, backwards-compatible args
    parser.add_argument("--prompt", default="upbeat pop", help="Text prompt/tags")
    parser.add_argument("--lyrics", default="", help="Lyrics in ACE Step format")
    parser.add_argument("--length", type=float, default=5, help="Audio length in seconds")
    parser.add_argument("--output", default="song.flac", help="Output audio filename")
    parser.add_argument(
        "--url",
        default="https://gcet--ace-step-flask-api-flask-app-dev.modal.run/generate",
        help="URL of the /generate endpoint",
    )

    # Full parameter set (all optional). If omitted, server-side defaults apply.
    parser.add_argument("--format", choices=["mp3", "ogg", "flac", "wav"], help="Output audio format")
    parser.add_argument("--audio-duration", type=float, help="Audio duration in seconds")
    parser.add_argument("--infer-step", type=int, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, help="Guidance scale")
    parser.add_argument("--scheduler-type", choices=["euler", "heun", "pingpong"], help="Scheduler type")
    parser.add_argument("--cfg-type", choices=["cfg", "apg", "cfg_star"], help="CFG type")
    parser.add_argument("--omega-scale", type=float, help="Granularity scale")
    parser.add_argument("--manual-seeds", help="Comma-separated seeds, e.g. '1,2,3'")
    parser.add_argument("--guidance-interval", type=float, help="Guidance interval [0,1]")
    parser.add_argument("--guidance-interval-decay", type=float, help="Guidance scale decay [0,1]")
    parser.add_argument("--min-guidance-scale", type=float, help="Min guidance scale")
    parser.add_argument("--use-erg-tag", choices=["true", "false"], help="Enable ERG for tags")
    parser.add_argument("--use-erg-lyric", choices=["true", "false"], help="Enable ERG for lyrics")
    parser.add_argument("--use-erg-diffusion", choices=["true", "false"], help="Enable ERG for diffusion")
    parser.add_argument("--oss-steps", help="Comma-separated optimal step indices")
    parser.add_argument("--guidance-scale-text", type=float, help="Guidance scale for text-only condition")
    parser.add_argument("--guidance-scale-lyric", type=float, help="Guidance scale for lyric-only condition")

    # Audio2Audio + LoRA
    parser.add_argument("--audio2audio-enable", choices=["true", "false"], help="Enable audio-to-audio")
    parser.add_argument("--ref-audio-strength", type=float, help="Reference audio strength [0,1]")
    parser.add_argument("--ref-audio-input", help="Path to reference audio (server-side accessible)")
    parser.add_argument("--lora-name-or-path", help="LoRA name or path")
    parser.add_argument("--lora-weight", type=float, help="LoRA weight")

    # Retake/Repaint/Edit/Extend
    parser.add_argument("--task", choices=["text2music", "retake", "repaint", "edit", "extend"], help="Task type")
    parser.add_argument("--retake-seeds", help="Comma-separated seeds for retake/repaint/extend")
    parser.add_argument("--retake-variance", type=float, help="Retake variance [0,1]")
    parser.add_argument("--repaint-start", type=float, help="Repaint start time (seconds)")
    parser.add_argument("--repaint-end", type=float, help="Repaint end time (seconds)")
    parser.add_argument("--src-audio-path", help="Source audio path for editing/retake/repaint/extend")
    parser.add_argument("--edit-target-prompt", help="Edit target tags")
    parser.add_argument("--edit-target-lyrics", help="Edit target lyrics")
    parser.add_argument("--edit-n-min", type=float, help="Edit n_min [0,1]")
    parser.add_argument("--edit-n-max", type=float, help="Edit n_max [0,1]")

    # Misc
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--save-path", help="Server-side save path (optional)")
    parser.add_argument("--debug", choices=["true", "false"], help="Enable debug mode")

    # Presets and test harness options
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), help="Use a predefined parameter set")
    parser.add_argument("--send-all-params", action="store_true", help="Send all known parameters with defaults (even if not set explicitly)")
    parser.add_argument(
        "--send-all-params-mode",
        choices=["plain", "audio2audio"],
        default="plain",
        help="Variant for --send-all-params: plain (no audio2audio) or audio2audio",
    )
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    # Test harness options
    parser.add_argument("--test", choices=["manual", "auto"], help="Run in test mode: manual prompts or automatic suite")
    parser.add_argument("--test-output-dir", help="Directory to store outputs for test mode")
    parser.add_argument("--dry-run", action="store_true", help="Print request(s) without sending")

    args = parser.parse_args()

    if args.list_presets:
        print("Available presets:")
        for name in sorted(PRESETS.keys()):
            print(f"- {name}")
        print("\nPreset details (JSON):")
        print(json.dumps(PRESETS, indent=2))
        return

    # Test modes
    if args.test == "auto":
        _run_auto_tests(args)
        return
    if args.test == "manual":
        _run_manual_test(args)
        return

    # Single request mode (default / backward-compatible)
    payload = _assemble_payload_from_args(args)
    output_path = Path(args.output)
    _send_request(args.url, payload, output_path, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()