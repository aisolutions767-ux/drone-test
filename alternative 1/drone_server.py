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

import os
import time
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

        # 2. Run Render Video (Orbit, Grid Scan, or path flythrough)
        print(f"[Mission Router] Rendering final drone video for {payload.location}...")
        render_cmd = build_render_cmd(payload)
        render_res = subprocess.run(
            render_cmd, cwd=str(DRONE_DIR),
            capture_output=True, text=True, timeout=600
        )

        video_path = DRONE_DIR / "drone_video.mp4"
        if render_res.returncode == 0 and video_path.exists():
            import time
            import re
            import shutil
            safe_location = re.sub(r'[^a-zA-Z0-9_-]', '_', payload.location)
            unique_name = f"mission_{safe_location}_{int(time.time())}.mp4"
            dest_path = DRONE_DIR / "videos" / unique_name
            shutil.copy2(video_path, dest_path)

            # Upload video to Gemini for cinematic review & fix recommendations (Alternative 1)
            gemini_key = os.getenv("GEMINI_API_KEY")
            gemini_review = None
            if gemini_key:
                try:
                    print(f"[Alternative 1] Uploading video {unique_name} to Gemini for cinematic review...")
                    from google import genai
                    from google.genai import types
                    
                    client = genai.Client(api_key=gemini_key)
                    
                    # Upload the video file
                    video_upload = client.files.upload(file=str(dest_path))
                    print(f"[Alternative 1] Video uploaded. Initial state: {video_upload.state.name}")
                    
                    # Wait for video processing
                    processing_timeout = 60
                    start_proc = time.time()
                    while video_upload.state.name == "PROCESSING":
                        if time.time() - start_proc > processing_timeout:
                            print("[Alternative 1] Video processing timed out on Gemini server.")
                            break
                        time.sleep(2)
                        video_upload = client.files.get(name=video_upload.name)
                        print(f"[Alternative 1] Video processing state: {video_upload.state.name}")
                        
                    if video_upload.state.name == "ACTIVE":
                        print("[Alternative 1] Video is active. Querying Gemini 2.0 Flash for cinematic analysis...")
                        prompt = (
                            "You are an expert cinematic drone pilot and director. Watch this drone flight video.\n"
                            "Evaluate the camera motion, smoothness, framing of the target, and lighting.\n"
                            "Provide 3 specific professional recommendations to improve the flight path or angles "
                            "to make the next take look like a premium hollywood movie drone shot.\n"
                            "Be direct, structured, and constructive."
                        )
                        
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[video_upload, prompt]
                        )
                        gemini_review = response.text.strip()
                        print("[Alternative 1] Gemini Video Review complete!")
                    else:
                        gemini_review = f"Video upload state is {video_upload.state.name}. Could not analyze."
                        
                    # Clean up the file from Gemini storage
                    client.files.delete(name=video_upload.name)
                    print("[Alternative 1] Cleaned up video file from Gemini cloud storage.")
                    
                except Exception as ex:
                    print(f"[Alternative 1] Failed to run Gemini Video Review: {ex}")
                    gemini_review = f"Error during Gemini Video Review: {str(ex)}"

            return {
                "status": "success",
                "message": "Drone mission complete!",
                "mode": payload.mode,
                "location": payload.location,
                "video": f"videos/{unique_name}",
                "local_path": str(dest_path.absolute()),
                "video_exists": True,
                "gemini_review": gemini_review
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



from fastapi.responses import StreamingResponse

@app.post("/generate_video_ai")
def generate_video_ai(payload: dict):
    def event_stream():
        yield "data: " + json.dumps({"status": "info", "message": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"}) + "\n\n"
        yield "data: " + json.dumps({"status": "info", "message": "🎬 [ALTERNATIVE 1] Full Video AI Pipeline"}) + "\n\n"
        yield "data: " + json.dumps({"status": "info", "message": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"}) + "\n\n"
        time.sleep(0.3)

        # ── STEP 1: Extract flight parameters from payload ──────────────
        yield "data: " + json.dumps({"status": "info", "message": "📋 STEP 1/5 — Extracting flight parameters..."}) + "\n\n"

        path_data = payload.get("path", [])
        location = payload.get("location", "Unknown Location")
        lat = payload.get("lat", 41.0)
        lon = payload.get("lon", 29.0)
        cam_range = payload.get("range", 350)
        pitch = payload.get("pitch", -25)
        heading = payload.get("heading", 45)
        mode = payload.get("mode", "single")

        # Determine render mode from path data
        if path_data and isinstance(path_data, list) and len(path_data) > 1:
            render_mode = "path"
            num_frames = max(36, len(path_data) * 12)  # 12 frames per waypoint for smooth motion
            yield "data: " + json.dumps({"status": "info", "message": f"   Mode: PATH fly-through ({len(path_data)} waypoints)"}) + "\n\n"
            yield "data: " + json.dumps({"status": "info", "message": f"   Location: {location}"}) + "\n\n"
            yield "data: " + json.dumps({"status": "info", "message": f"   Frames to render: {num_frames}"}) + "\n\n"
            # Use first waypoint as start position
            if "lat" in path_data[0]:
                lat = path_data[0]["lat"]
                lon = path_data[0]["lon"]
        else:
            render_mode = "single"
            num_frames = 72  # Full 360° orbit
            yield "data: " + json.dumps({"status": "info", "message": f"   Mode: SINGLE orbit (360° sweep)"}) + "\n\n"
            yield "data: " + json.dumps({"status": "info", "message": f"   Location: {location} ({lat}, {lon})"}) + "\n\n"
            yield "data: " + json.dumps({"status": "info", "message": f"   Frames to render: {num_frames}"}) + "\n\n"

        yield "data: " + json.dumps({"status": "info", "message": f"   Camera: range={cam_range}m, pitch={pitch}°, heading={heading}°"}) + "\n\n"
        time.sleep(0.3)

        # ── STEP 2: Render full video via Playwright + OpenCV ───────────
        yield "data: " + json.dumps({"status": "info", "message": "🎥 STEP 2/5 — Rendering drone flight video via Playwright..."}) + "\n\n"
        yield "data: " + json.dumps({"status": "info", "message": "   Launching headless Chromium browser..."}) + "\n\n"
        yield "data: " + json.dumps({"status": "info", "message": "   Loading Cesium 3D globe + Google Photorealistic tiles..."}) + "\n\n"

        render_script = DRONE_DIR / "drone_render_video.py"
        if not render_script.exists():
            render_script = DRONE_DIR.parent / "drone_render_video.py"

        if not render_script.exists():
            yield "data: " + json.dumps({"status": "error", "message": "❌ Error: drone_render_video.py not found!"}) + "\n\n"
            return

        # Build render command
        cmd = [
            sys.executable, str(render_script),
            "--location", str(location),
            "--mode", render_mode,
            "--lat", str(lat), "--lon", str(lon),
            "--range", str(int(cam_range)),
            "--pitch", str(int(pitch)),
            "--heading", str(int(heading)),
            "--frames", str(num_frames),
            "--fps", "24"
        ]

        if render_mode == "path" and path_data:
            # Clean path data — only keep lat/lon for the render script
            clean_path = []
            for pt in path_data:
                if isinstance(pt, dict) and "lat" in pt and "lon" in pt:
                    clean_path.append({"lat": pt["lat"], "lon": pt["lon"]})
            if clean_path:
                cmd.extend(["--path", json.dumps(clean_path)])

        yield "data: " + json.dumps({"status": "info", "message": f"   Command: drone_render_video.py --mode {render_mode} --frames {num_frames}"}) + "\n\n"
        yield "data: " + json.dumps({"status": "info", "message": "   ⏳ Rendering frames... (this may take 30-120 seconds)"}) + "\n\n"

        try:
            # Run render with real-time output streaming
            render_proc = subprocess.Popen(
                cmd, cwd=str(DRONE_DIR),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )

            output_lines = []
            for line in iter(render_proc.stdout.readline, ""):
                line = line.strip()
                if line:
                    output_lines.append(line)
                    # Stream key progress lines to the frontend
                    if any(kw in line.lower() for kw in ["rendered", "frame", "compiling", "video ready", "error", "warning", "loaded", "mode"]):
                        yield "data: " + json.dumps({"status": "info", "message": f"   📸 {line}"}) + "\n\n"

            render_proc.wait(timeout=300)

            if render_proc.returncode != 0:
                error_output = "\n".join(output_lines[-10:])
                yield "data: " + json.dumps({"status": "error", "message": f"❌ Render failed (exit code {render_proc.returncode})"}) + "\n\n"
                yield "data: " + json.dumps({"status": "error", "message": f"   Last output: {error_output[-500:]}"}) + "\n\n"
                return

        except subprocess.TimeoutExpired:
            render_proc.kill()
            yield "data: " + json.dumps({"status": "error", "message": "❌ Render timed out after 300 seconds."}) + "\n\n"
            return
        except Exception as render_err:
            yield "data: " + json.dumps({"status": "error", "message": f"❌ Render exception: {str(render_err)}"}) + "\n\n"
            return

        # ── STEP 3: Locate rendered video ───────────────────────────────
        yield "data: " + json.dumps({"status": "info", "message": "📂 STEP 3/5 — Locating rendered video file..."}) + "\n\n"

        # Check multiple locations for the output video
        target_video = None
        search_paths = [
            DRONE_DIR / "drone_video.mp4",
            DRONE_DIR.parent / "drone_video.mp4",
            Path("drone_video.mp4"),
        ]
        # Also check videos/ subdirectories
        for vdir in [DRONE_DIR / "videos", DRONE_DIR.parent / "videos"]:
            if vdir.exists():
                mp4s = sorted(vdir.glob("*.mp4"), key=os.path.getmtime)
                if mp4s:
                    search_paths.insert(0, mp4s[-1])

        for vp in search_paths:
            if vp.exists():
                target_video = vp
                break

        if not target_video:
            yield "data: " + json.dumps({"status": "error", "message": "❌ Error: Video render completed but output .mp4 not found!"}) + "\n\n"
            return

        size_mb = target_video.stat().st_size / (1024 * 1024)
        yield "data: " + json.dumps({"status": "info", "message": f"   ✅ Found: {target_video.name} ({size_mb:.1f} MB)"}) + "\n\n"
        time.sleep(0.3)

        # ── STEP 4: Upload to Gemini & Get AI Analysis ──────────────────
        yield "data: " + json.dumps({"status": "info", "message": "🤖 STEP 4/5 — Uploading video to Google Gemini AI..."}) + "\n\n"

        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            yield "data: " + json.dumps({"status": "error", "message": "❌ Error: GEMINI_API_KEY not set in .env or environment."}) + "\n\n"
            return

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            yield "data: " + json.dumps({"status": "info", "message": "   Connected to Gemini API."}) + "\n\n"

            yield "data: " + json.dumps({"status": "info", "message": f"   Uploading {size_mb:.1f} MB video to Gemini cloud storage..."}) + "\n\n"
            video_upload = client.files.upload(file=str(target_video))
            yield "data: " + json.dumps({"status": "info", "message": f"   Upload complete. File ref: {video_upload.name}"}) + "\n\n"

            # Wait for Gemini to index the video
            start_proc = time.time()
            processing_timeout = 120
            while video_upload.state.name == "PROCESSING":
                elapsed = int(time.time() - start_proc)
                if elapsed > processing_timeout:
                    yield "data: " + json.dumps({"status": "error", "message": "❌ Video processing timed out on Gemini server."}) + "\n\n"
                    break
                yield "data: " + json.dumps({"status": "info", "message": f"   ⏳ Gemini indexing video... ({elapsed}s elapsed)"}) + "\n\n"
                time.sleep(3)
                video_upload = client.files.get(name=video_upload.name)

            if video_upload.state.name != "ACTIVE":
                yield "data: " + json.dumps({"status": "error", "message": f"❌ Video state invalid: {video_upload.state.name}"}) + "\n\n"
                return

            yield "data: " + json.dumps({"status": "info", "message": "   ✅ Video indexed and ready for analysis."}) + "\n\n"
            time.sleep(0.3)

            # ── STEP 5: AI Cinematic Analysis & Enhancement ─────────────
            yield "data: " + json.dumps({"status": "info", "message": "🎬 STEP 5/5 — Generating AI cinematic analysis..."}) + "\n\n"
            yield "data: " + json.dumps({"status": "info", "message": "   Sending to Gemini 2.0 Flash for deep video analysis..."}) + "\n\n"

            prompt = (
                "You are a world-class cinematic drone videographer and post-production director. "
                "Watch this drone flight video carefully.\n\n"
                "Provide a comprehensive analysis with these sections:\n\n"
                "## 🎥 Flight Quality Assessment\n"
                "Rate the overall smoothness, camera motion, and composition (1-10 score).\n\n"
                "## 🔧 Specific Issues Found\n"
                "List any jerky movements, poor framing, bad angles, or visual artifacts.\n\n"
                "## ✨ Enhancement Recommendations\n"
                "Provide 5 specific, actionable improvements:\n"
                "1. Camera angle & movement improvements\n"
                "2. Speed/pacing adjustments\n"
                "3. Composition & framing fixes\n"
                "4. Suggested color grading & post-production effects\n"
                "5. Cinematic techniques to apply (dolly zoom, reveal shots, etc.)\n\n"
                "## 🎬 AI Video Enhancement Prompt\n"
                "Write a detailed prompt that could be fed to an AI video enhancer (like Runway, Sora, or Kling) "
                "to transform this raw 3D simulation into a photorealistic cinematic drone shot. "
                "Include descriptions of realistic lighting, shadows, textures, weather, "
                "atmospheric haze, and motion blur that should be added.\n\n"
                "Be specific, professional, and constructive."
            )

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[video_upload, prompt]
            )

            yield "data: " + json.dumps({
                "status": "success",
                "message": "✅ AI Cinematic Analysis Complete!",
                "result": response.text.strip()
            }) + "\n\n"

            # Cleanup
            try:
                client.files.delete(name=video_upload.name)
                yield "data: " + json.dumps({"status": "info", "message": "🧹 Cleaned up video from Gemini storage."}) + "\n\n"
            except Exception:
                pass

        except Exception as e:
            yield "data: " + json.dumps({"status": "error", "message": f"❌ AI Processing error: {str(e)}"}) + "\n\n"

        yield "data: " + json.dumps({"status": "info", "message": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"}) + "\n\n"
        yield "data: " + json.dumps({"status": "info", "message": "Pipeline complete. All steps finished."}) + "\n\n"

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
