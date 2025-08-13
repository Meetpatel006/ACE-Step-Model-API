import argparse
import base64
import json
from pathlib import Path
import sys

import requests


def main():
    parser = argparse.ArgumentParser(description="Test Modal ACEStep API /generate route")

    # Endpoint
    parser.add_argument(
        "--url",
        default="https://gcet--ace-step-flask-api-flask-app-dev.modal.run/generate",
        help="Modal API /generate endpoint",
    )
    parser.add_argument("--output", default="modal_song.wav", help="Output audio file path")
    parser.add_argument("--dry-run", action="store_true", help="Print payload without sending")

    # Required feature params (no audio2audio here)
    parser.add_argument("--audio-duration", type=float, default=10.0, help="Audio length in seconds")
    parser.add_argument(
        "--prompt",
        default="funk, pop, energetic",
        help=(
            "Tags/description/scene, comma-separated. "
            "Support tags, descriptions, and scene. Use commas to separate different tags."
        ),
    )
    parser.add_argument("--lyrics", default="[instrumental]", help="Lyrics text (supports [verse]/[chorus] markers)")

    parser.add_argument("--infer-step", type=int, default=60, help="Number of diffusion steps (1..200)")
    parser.add_argument("--guidance-scale", type=float, default=15.0, help="CFG guidance scale")
    parser.add_argument("--guidance-scale-text", type=float, default=0.0, help="Guidance for text-only condition")
    parser.add_argument("--guidance-scale-lyric", type=float, default=0.0, help="Guidance for lyric-only condition")

    parser.add_argument(
        "--scheduler-type",
        choices=["euler", "heun"],
        default="euler",
        help="Scheduler type (subset requested: euler, heun)",
    )
    parser.add_argument(
        "--cfg-type",
        choices=["cfg", "apg", "cfg_star"],
        default="apg",
        help="CFG type",
    )

    parser.add_argument("--omega-scale", type=float, default=10.0, help="Granularity scale")
    parser.add_argument("--guidance-interval", type=float, default=0.5, help="Guidance window fraction [0..1]")
    parser.add_argument("--guidance-interval-decay", type=float, default=0.0, help="Guidance decay across window [0..1]")
    parser.add_argument("--min-guidance-scale", type=float, default=3.0, help="Min guidance scale for decay")

    parser.add_argument(
        "--oss-steps",
        default=None,
        help="Comma-separated optimal step indices (e.g. '16,29,52') or leave empty",
    )

    args = parser.parse_args()

    payload = {
        # core
        "audio_duration": args.audio_duration,
        "prompt": args.prompt,
        "lyrics": args.lyrics,
        # basics
        "infer_step": args.infer_step,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_text": args.guidance_scale_text,
        "guidance_scale_lyric": args.guidance_scale_lyric,
        #advance setting
        "scheduler_type": args.scheduler_type,
        "cfg_type": args.cfg_type,
        "omega_scale": args.omega_scale,
        "guidance_interval": args.guidance_interval,
        "guidance_interval_decay": args.guidance_interval_decay,
        "min_guidance_scale": args.min_guidance_scale,
        "oss_steps": args.oss_steps,
    }

    if args.dry_run:
        print(json.dumps({"url": args.url, "payload": payload}, indent=2))
        sys.exit(0)

    resp = requests.post(args.url, json=payload)
    if resp.status_code != 200:
        print("Request failed:", resp.status_code, resp.text)
        sys.exit(1)

    data = resp.json()
    audio_b64 = data.get("audio_base64")
    if not audio_b64:
        print("Response missing audio data")
        sys.exit(1)

    audio_bytes = base64.b64decode(audio_b64)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    print("Saved:", str(out_path))


if __name__ == "__main__":
    main()
