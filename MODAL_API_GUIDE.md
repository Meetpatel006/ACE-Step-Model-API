## ACEStep Modal API guide

This guide documents the API exposed by the Modal app defined in `modal_app.py` and provides a small tester script for the `/generate` route.

### Endpoints (from `modal_app.py`)

- POST `/generate` (two implementations in the app: FastAPI endpoint via `@modal.fastapi_endpoint` and WSGI Flask app route)
  - Request JSON body fields (relevant for the requested feature subset):
    - `audio_duration` (or `length`): number (seconds). If `audio_duration` is absent, `length` is supported for backward compatibility.
    - `prompt`: string. Tags/description/scene, comma separated.
    - `lyrics`: string. Supports markers like `[verse]`, `[chorus]`, `[bridge]`.
    - `infer_step`: int (1..200). Number of diffusion steps.
    - `guidance_scale`: float. CFG guidance scale.
    - `guidance_scale_text`: float. Guidance scale for text-only condition.
    - `guidance_scale_lyric`: float. Guidance scale for lyric-only condition.
    - `scheduler_type`: "euler" | "heun" | "pingpong".
    - `cfg_type`: "cfg" | "apg" | "cfg_star".
    - `omega_scale`: float. Granularity scale.
    - `guidance_interval`: float [0..1].
    - `guidance_interval_decay`: float [0..1].
    - `min_guidance_scale`: float.
    - `oss_steps`: string like "16, 29, 52" or int[] list.
  - Response JSON body:
    - `{ "audio_base64": "..." }`

- GET `/health` (Flask app)
  - Returns `{ "status": "healthy" }`

Note: The Modal app also accepts many advanced parameters (audio2audio, LoRA, repaint/edit/extend). See `_extract_generation_params` in `modal_app.py` for the full list.

### Tester script
- `modal_api_tester.py` calls the `/generate` endpoint with the parameters listed above and saves a WAV.

Run it:
```bash
python modal_api_tester.py \
  --url https://gcet--ace-step-flask-api-flask-app-dev.modal.run/generate \
  --audio-duration 10 \
  --prompt "funk, pop, energetic" \
  --lyrics "[instrumental]" \
  --infer-step 60 \
  --guidance-scale 15 \
  --guidance-scale-text 0.0 \
  --guidance-scale-lyric 0.0 \
  --scheduler-type euler \
  --cfg-type apg \
  --omega-scale 10 \
  --guidance-interval 0.5 \
  --guidance-interval-decay 0.0 \
  --min-guidance-scale 3.0 \
  --oss-steps "16, 29, 52" \
  --output modal_song.wav
```

### Tips
- If your Modal app is the WSGI Flask route, the base path is the same (`/generate`).
- Large values for `infer_step` increase latency and cost; start small for quick tests.
- If using `oss_steps`, the server will adjust internal timesteps accordingly.
- The service returns base64; the tester decodes to WAV. Change `--output` extension to save a different extension if you change `format` server-side.
