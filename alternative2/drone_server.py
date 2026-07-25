# /// script
# dependencies = [
#   "fastapi>=0.110.0",
#   "uvicorn>=0.29.0",
#   "python-dotenv>=1.0.0",
#   "playwright>=1.40.0",
#   "google-genai>=0.1.1",
#   "click>=8.1.0",
#   "pillow>=10.0.0",
#   "opencv-python>=4.8.0.0",
#   "requests>=2.31.0",
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

# Ensure videos and screenshots directories exist
(DRONE_DIR / "videos").mkdir(parents=True, exist_ok=True)
(DRONE_DIR / "screenshots").mkdir(parents=True, exist_ok=True)

# Mount static directories for media access
from fastapi.staticfiles import StaticFiles
app.mount("/videos", StaticFiles(directory=str(DRONE_DIR / "videos")), name="videos")
app.mount("/screenshots", StaticFiles(directory=str(DRONE_DIR / "screenshots")), name="screenshots")


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
    import shutil
    has_xvfb = shutil.which("xvfb-run") is not None
    cmd = []
    if has_xvfb:
        cmd += ["xvfb-run", "-a", "--server-args=-screen 0 1280x720x24"]
    cmd += [
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
    import shutil
    has_xvfb = shutil.which("xvfb-run") is not None
    cmd = []
    if has_xvfb:
        cmd += ["xvfb-run", "-a", "--server-args=-screen 0 1280x720x24"]
    cmd += [
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
        if result.returncode == 0 and video_path.exists():
            import time
            import re
            import shutil
            safe_location = re.sub(r'[^a-zA-Z0-9_-]', '_', payload.location)
            unique_name = f"mission_{safe_location}_{int(time.time())}.mp4"
            dest_path = DRONE_DIR / "videos" / unique_name
            shutil.copy2(video_path, dest_path)
            return {
                "status": "done",
                "stdout": result.stdout[-3000:],
                "stderr": result.stderr[-1000:],
                "returncode": result.returncode,
                "location": payload.location,
                "mode": payload.mode,
                "video": f"videos/{unique_name}",
                "local_path": str(dest_path.absolute()),
                "video_exists": True,
            }
        else:
            return {
                "status": "error",
                "stdout": result.stdout[-3000:],
                "stderr": result.stderr[-1000:],
                "returncode": result.returncode,
                "location": payload.location,
                "mode": payload.mode,
                "video": None,
                "video_exists": False,
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

        # 2. Extract keyframe screenshots from the payload path and send them to Gemini (Alternative 2)
        gemini_key = os.getenv("GEMINI_API_KEY")
        gemini_video_script = None
        
        # Check if the payload has path waypoints containing screenshots (from 3D Record Mode)
        keyframes = payload.path if payload.path else []
        
        if gemini_key and keyframes:
            try:
                import base64
                from google import genai
                from google.genai import types
                
                print(f"[Alternative 2] Uploading {len(keyframes)} keyframe screenshots to Gemini...")
                client = genai.Client(api_key=gemini_key)
                
                contents = []
                # Append each keyframe screenshot to the contents
                for idx, kf in enumerate(keyframes):
                    img_data = kf.get("screenshot")
                    if img_data and "," in img_data:
                        base64_data = img_data.split(",")[1]
                        image_bytes = base64.b64decode(base64_data)
                        contents.append(
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/jpeg",
                                    data=image_bytes
                                )
                            )
                        )
                        print(f"[Alternative 2] Loaded screenshot for keyframe {idx + 1}")
                
                if contents:
                    prompt = (
                        f"You are an AI video generation director. You are given {len(contents)} sequential keyframe screenshots "
                        f"representing a flight path over: '{payload.location}'.\n"
                        "Your task is to analyze these screenshots and generate a detailed video prompt script for AI video generators (like Sora or Runway Gen-3).\n"
                        "For each keyframe transition, output a detailed prompt describing the camera angle, motion, realistic lighting, shadows, "
                        "and texture enhancements (e.g. realistic reflections, detailed foliage, weather, cinematic look) needed to turn these "
                        "simple 3D viewports into a highly realistic drone shot video.\n"
                        "Respond with a clear, structured Markdown timeline script."
                    )
                    contents.append(types.Part(text=prompt))
                    
                    print("[Alternative 2] Sending frames to Gemini 2.0 Flash...")
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=contents
                    )
                    gemini_video_script = response.text.strip()
                    print("[Alternative 2] Gemini Video Script generation complete!")
                else:
                    gemini_video_script = "No valid keyframe screenshots found in the path payload."
                    
            except Exception as ex:
                print(f"[Alternative 2] Failed to generate video script: {ex}")
                gemini_video_script = f"Error during Gemini Video Script generation: {str(ex)}"
        else:
            gemini_video_script = "Gemini key or keyframe path not present. Skipping script generation."

        # Return a simulated success response containing the path data and the AI video script
        return {
            "status": "success",
            "message": "Drone path processed successfully via AI Frame Generator!",
            "mode": payload.mode,
            "location": payload.location,
            "video": None,
            "video_exists": False,
            "path": payload.path,
            "gemini_video_script": gemini_video_script
        }

    except subprocess.TimeoutExpired as te:
        raise HTTPException(status_code=504, detail=f"Mission execution timeout: {str(te)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



from fastapi.responses import StreamingResponse
import base64

@app.post("/generate_video_ai")
def generate_video_ai(payload: dict):
    def event_stream():
        yield "data: " + json.dumps({"status": "info", "message": "[Alternative 2 Server] Initializing Frame-by-Frame AI workflow..."}) + "\n\n"
        time.sleep(1)
        
        keyframes = payload.get("path", [])
        if not keyframes:
            yield "data: " + json.dumps({"status": "error", "message": "Error: No keyframe coordinates found. Please record a path or waypoints first."}) + "\n\n"
            return
            
        yield "data: " + json.dumps({"status": "info", "message": f"Successfully unpacked path with {len(keyframes)} keyframe nodes."}) + "\n\n"
        time.sleep(0.5)
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            yield "data: " + json.dumps({"status": "error", "message": "Error: GEMINI_API_KEY environment variable is not set."}) + "\n\n"
            return
            
        try:
            from google import genai
            from google.genai import types
            
            yield "data: " + json.dumps({"status": "info", "message": "Connecting to Google Gemini API..."}) + "\n\n"
            client = genai.Client(api_key=gemini_key)
            
            contents = []
            valid_images_count = 0
            
            for idx, kf in enumerate(keyframes):
                img_data = kf.get("screenshot")
                if img_data and "," in img_data:
                    yield "data: " + json.dumps({"status": "info", "message": f"Processing screenshot for keyframe {idx + 1}... decoding Base64 payload."}) + "\n\n"
                    try:
                        base64_data = img_data.split(",")[1]
                        image_bytes = base64.b64decode(base64_data)
                        contents.append(
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/jpeg",
                                    data=image_bytes
                                )
                            )
                        )
                        valid_images_count += 1
                        time.sleep(0.1)
                    except Exception as dec_err:
                        yield "data: " + json.dumps({"status": "warning", "message": f"Warning: Failed to decode frame {idx + 1}: {str(dec_err)}"}) + "\n\n"
            
            if not contents:
                yield "data: " + json.dumps({"status": "error", "message": "Error: No valid base64 screenshots were attached to the flight path."}) + "\n\n"
                return
                
            yield "data: " + json.dumps({"status": "info", "message": f"Sending {valid_images_count} frames and telemetry constraints to Gemini 2.0 Flash..."}) + "\n\n"
            
            prompt = (
                f"You are an AI video generation director. You are given {valid_images_count} sequential keyframe screenshots "
                f"representing a flight path over: '{payload.get('location', 'Selected Coordinates')}'.\n"
                "Your task is to analyze these screenshots and generate a detailed video prompt script for AI video generators (like Sora or Runway Gen-3).\n"
                "For each keyframe transition, output a detailed prompt describing the camera angle, motion, realistic lighting, shadows, "
                "and texture enhancements (e.g. realistic reflections, detailed foliage, weather, cinematic look) needed to turn these "
                "simple 3D viewports into a highly realistic drone shot video.\n"
                "Respond with a clear, structured Markdown timeline script."
            )
            contents.append(types.Part(text=prompt))
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents
            )
            
            yield "data: " + json.dumps({
                "status": "success",
                "message": "AI storyboard and prompt timeline generated successfully!",
                "result": response.text.strip()
            }) + "\n\n"
            
        except Exception as e:
            yield "data: " + json.dumps({"status": "error", "message": f"Processing exception occurred: {str(e)}"}) + "\n\n"
            
    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
