# /// script
# dependencies = [
#   "fastapi>=0.110.0",
#   "uvicorn>=0.29.0",
#   "python-dotenv>=1.0.0",
# ]
# ///

"""
drone_server.py
---------------
Local HTTP API server that n8n calls via HTTP Request nodes.
Replaces shell Execute Command nodes (which are community-only).

Run with:
  uv run drone_server.py

Endpoints:
  POST /screenshot   - Run AI screenshot review (any mode)
  POST /render       - Render full video (any mode)
  GET  /health       - Health check
"""

import subprocess
import sys
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any

app = FastAPI(title="Virtual Drone Mission API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DRONE_DIR = Path(__file__).parent


class MissionPayload(BaseModel):
    location: str
    mode: str                        # "single" | "area" | "path"
    lat: Optional[float] = None
    lon: Optional[float] = None
    range: Optional[int] = 350
    pitch: Optional[int] = -25
    heading: Optional[int] = 45
    frames: Optional[int] = 72
    fps: Optional[int] = 24
    polygon: Optional[List[Any]] = None
    path: Optional[List[Any]] = None
    extruded_height: Optional[float] = None
    camera_lat: Optional[float] = None
    camera_lon: Optional[float] = None
    camera_height: Optional[float] = None
    camera_heading: Optional[float] = None
    camera_pitch: Optional[float] = None
    camera_roll: Optional[float] = None
    building_height: Optional[float] = None
    surface_elevation: Optional[float] = None
    terrain_elevation: Optional[float] = None
    staging_screenshot: Optional[str] = None


def save_staging_screenshot(payload_screenshot: Optional[str]) -> Optional[str]:
    if payload_screenshot and "," in payload_screenshot:
        try:
            import base64
            header, data = payload_screenshot.split(",", 1)
            img_data = base64.b64decode(data)
            out_dir = DRONE_DIR / "screenshots"
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / "staging_input.jpg"
            out_path.write_bytes(img_data)
            print(f"[Server] Saved staging screenshot to: {out_path}")
            return str(out_path)
        except Exception as e:
            print(f"[Server Warning] Failed to decode staging screenshot: {e}")
    return None


def build_screenshot_cmd(payload: MissionPayload, staging_img_path: Optional[str] = None) -> list:
    import sys
    cmd = [
        sys.executable,
        str(DRONE_DIR / "drone_capture_screenshots.py"),
        "--location", payload.location,
        "--mode", payload.mode,
        "--range", str(payload.range),
        "--pitch", str(payload.pitch),
        "--heading", str(payload.heading),
    ]
    if payload.mode == "single" and payload.lat and payload.lon:
        cmd += ["--lat", str(payload.lat), "--lon", str(payload.lon)]
        if payload.camera_lat is not None:
            cmd += ["--camera-lat", str(payload.camera_lat)]
        if payload.camera_lon is not None:
            cmd += ["--camera-lon", str(payload.camera_lon)]
        if payload.camera_height is not None:
            cmd += ["--camera-height", str(payload.camera_height)]
        if payload.camera_heading is not None:
            cmd += ["--camera-heading", str(payload.camera_heading)]
        if payload.camera_pitch is not None:
            cmd += ["--camera-pitch", str(payload.camera_pitch)]
        if payload.camera_roll is not None:
            cmd += ["--camera-roll", str(payload.camera_roll)]
        if payload.building_height is not None:
            cmd += ["--building-height", str(payload.building_height)]
        if payload.surface_elevation is not None:
            cmd += ["--surface-elevation", str(payload.surface_elevation)]
        if payload.terrain_elevation is not None:
            cmd += ["--terrain-elevation", str(payload.terrain_elevation)]
        if staging_img_path:
            cmd += ["--staging-screenshot", staging_img_path]
    elif payload.mode == "area" and payload.polygon:
        cmd += ["--polygon", json.dumps(payload.polygon)]
    elif payload.mode == "path" and payload.path:
        cmd += ["--path", json.dumps(payload.path)]
    return cmd


def build_render_cmd(payload: MissionPayload) -> list:
    import sys
    cmd = [
        sys.executable,
        str(DRONE_DIR / "drone_render_video.py"),
        "--location", payload.location,
        "--mode", payload.mode,
        "--range", str(payload.range),
        "--pitch", str(payload.pitch),
        "--heading", str(payload.heading),
        "--frames", str(payload.frames),
        "--fps", str(payload.fps),
    ]
    if payload.mode == "single" and payload.lat and payload.lon:
        cmd += ["--lat", str(payload.lat), "--lon", str(payload.lon)]
    elif payload.mode == "area" and payload.polygon:
        cmd += ["--polygon", json.dumps(payload.polygon)]
    elif payload.mode == "path" and payload.path:
        cmd += ["--path", json.dumps(payload.path)]
    return cmd


@app.get("/health")
def health():
    return {"status": "ok", "server": "Virtual Drone Mission API v1.0"}


@app.post("/screenshot")
def run_screenshot(payload: MissionPayload):
    staging_path = save_staging_screenshot(payload.staging_screenshot)
    cmd = build_screenshot_cmd(payload, staging_path)
    try:
        result = subprocess.run(
            cmd, cwd=str(DRONE_DIR),
            capture_output=True, text=True, timeout=120
        )
        return {
            "status": "done" if result.returncode == 0 else "error",
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-1000:],
            "returncode": result.returncode,
            "location": payload.location,
            "mode": payload.mode,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Screenshot timeout (120s)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/render")
def run_render(payload: MissionPayload):
    cmd = build_render_cmd(payload)
    try:
        result = subprocess.run(
            cmd, cwd=str(DRONE_DIR),
            capture_output=True, text=True, timeout=600
        )
        video_path = DRONE_DIR / "drone_video.mp4"
        return {
            "status": "done" if result.returncode == 0 else "error",
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-1000:],
            "returncode": result.returncode,
            "location": payload.location,
            "mode": payload.mode,
            "video": str(video_path) if video_path.exists() else None,
            "video_exists": video_path.exists(),
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Render timeout (600s)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/mission")
def run_mission(payload: MissionPayload):
    # 1. Run Screenshot Staging / Review (Gemini Vision Loop)
    staging_path = save_staging_screenshot(payload.staging_screenshot)
    screenshot_cmd = build_screenshot_cmd(payload, staging_path)
    try:
        print(f"[Mission Router] Triggering AI screenshot framing review for {payload.location}...")
        shot_res = subprocess.run(
            screenshot_cmd, cwd=str(DRONE_DIR),
            capture_output=True, text=True, timeout=120
        )
        if shot_res.returncode != 0:
            return {
                "status": "error",
                "stage": "screenshot_review",
                "stderr": shot_res.stderr[-1000:],
                "stdout": shot_res.stdout[-2000:]
            }

        # Parse optimized staging values from stdout
        stdout_lines = shot_res.stdout.split("\n")
        try:
            start_parse = False
            json_lines = []
            for line in stdout_lines:
                if "Final Approved Camera Params" in line or "Final Approved Camera Parameters" in line:
                    start_parse = True
                    continue
                if start_parse:
                    json_lines.append(line)
            if json_lines:
                approved_params = json.loads("".join(json_lines).strip())
                payload.lat = approved_params.get("lat", payload.lat)
                payload.lon = approved_params.get("lon", payload.lon)
                payload.range = int(approved_params.get("range", payload.range))
                payload.pitch = int(approved_params.get("pitch", payload.pitch))
                payload.heading = int(approved_params.get("heading", payload.heading))
                print(f"[Mission Router] AI-Optimized Staging applied: {approved_params}")
        except Exception as json_err:
            print(f"[Mission Router] Warning: Could not parse AI params: {json_err}")

        # 2. Run Render Video (Orbit, Grid Scan, or path flythrough)
        print(f"[Mission Router] Rendering final drone video for {payload.location}...")
        render_cmd = build_render_cmd(payload)
        render_res = subprocess.run(
            render_cmd, cwd=str(DRONE_DIR),
            capture_output=True, text=True, timeout=600
        )

        video_path = DRONE_DIR / "drone_video.mp4"
        if render_res.returncode == 0 and video_path.exists():
            return {
                "status": "success",
                "message": "Drone mission complete!",
                "mode": payload.mode,
                "location": payload.location,
                "video": str(video_path.absolute()),
                "video_exists": True
            }
        else:
            return {
                "status": "error",
                "stage": "video_render",
                "stderr": render_res.stderr[-1000:],
                "stdout": render_res.stdout[-2000:]
            }

    except subprocess.TimeoutExpired as te:
        raise HTTPException(status_code=504, detail=f"Mission execution timeout: {str(te)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/")
def serve_index():
    return FileResponse(DRONE_DIR / "index.html")


@app.get("/config.js")
def serve_config():
    try:
        config_path = DRONE_DIR / "config.js"
        if config_path.exists():
            content = config_path.read_text(encoding="utf-8")
            import os
            
            # Check if cloud environment variables are present
            gemini = os.getenv("GEMINI_API_KEY")
            openrouter = os.getenv("OPENROUTER_API_KEY")
            cesium = os.getenv("CESIUM_ION_TOKEN")
            
            # Replace placeholders in config.js dynamically with the environment variables
            if gemini and "__GEMINI_API_KEY__" in content:
                content = content.replace("__GEMINI_API_KEY__", gemini)
            if openrouter and "__OPENROUTER_API_KEY__" in content:
                content = content.replace("__OPENROUTER_API_KEY__", openrouter)
            if cesium and "__CESIUM_ION_TOKEN__" in content:
                content = content.replace("__CESIUM_ION_TOKEN__", cesium)
                
            from fastapi.responses import Response
            return Response(content=content, media_type="application/javascript")
    except Exception as e:
        print(f"[Server Warning] Failed to dynamically process config.js: {e}")
    return FileResponse(DRONE_DIR / "config.js")


@app.get("/viewer.html")
def serve_viewer():
    return FileResponse(DRONE_DIR / "viewer.html")



if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8765))
    print(f"Starting Drone Mission API on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
