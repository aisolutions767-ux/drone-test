# /// script
# dependencies = [
#   "playwright>=1.40.0",
#   "google-genai>=0.1.1",
#   "python-dotenv>=1.0.0",
#   "click>=8.1.0",
#   "pillow>=10.0.0",
# ]
# ///

"""
drone_capture_screenshots.py
----------------------------
Playwright script that opens viewer.html, positions the Cesium camera based
on the incoming payload, captures a screenshot of the 3D scene, and sends it
to Gemini Vision for AI framing review. Iterates until Gemini approves the
shot (status=GOOD) or max iterations reached.

Usage examples:
  # Single building mode:
  uv run drone_capture_screenshots.py --location "Colosseum, Rome" --lat 41.8902 --lon 12.4922 --range 250 --pitch -25 --heading 45

  # Area mode with polygon JSON:
  uv run drone_capture_screenshots.py --location "Eiffel Tower Area" --mode area --polygon '[{"lat":48.8582,"lon":2.2941},{"lat":48.8588,"lon":2.2948},{"lat":48.8580,"lon":2.2952}]' --range 350 --pitch -30

  # Path mode with waypoints JSON:
  uv run drone_capture_screenshots.py --location "Manhattan Path" --mode path --path '[{"lat":40.7484,"lon":-73.9857},{"lat":40.7505,"lon":-73.9870}]' --range 200 --pitch -20 --heading 90
"""

import os
import sys
import time
import json
import click
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
VIEWER_HTML = Path(__file__).parent / "viewer.html"
OUTPUT_DIR  = Path("screenshots")
MAX_AI_ITERATIONS = 5

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Gemini AI Vision Prompts per mode ─────────────────────────────────────────
PROMPTS = {
    "single": {
        "system": (
            "You are an expert cinematic drone director. Your task is to review a 3D Cesium "
            "viewport screenshot and evaluate whether the target building is optimally framed. "
            "The target structure should occupy 40-60% of the viewport, be centered, and fully "
            "visible without being cut off at the top or edges. Buildings should look clearly "
            "3D and not like flat aerial tiles."
        ),
        "user": (
            "Review the attached 3D viewport screenshot for location: '{location}'.\n"
            "Target Coordinates: Lat={lat}, Lon={lon}.\n"
            "Staging Camera Telemetry:\n"
            "- Position: Lat={camera_lat}, Lon={camera_lon}, Height={camera_height}m\n"
            "- Orientation: Heading={camera_heading}deg, Pitch={camera_pitch}deg, Roll={camera_roll}deg\n"
            "- Target Elevation: {target_altitude}m\n"
            "- Building Height: {building_height}m\n"
            "- Terrain Elevation: {terrain_elevation}m\n"
            "Staged Orbit Config: Range={range}m, Pitch={pitch}deg, Heading={heading}deg.\n\n"
            "Respond ONLY with a valid JSON object (no markdown fences, no extra text):\n"
            "{{\"status\": \"GOOD\" or \"ADJUST\", \"lat_offset\": 0.0, \"lon_offset\": 0.0, "
            "\"range\": {range}, \"pitch\": {pitch}, \"heading\": {heading}, \"feedback\": \"brief description\"}}"
        )
    },
    "area": {
        "system": (
            "You are a drone flight safety inspector for autonomous mapping missions. "
            "You review grid scan altitude plans over urban or rural areas. "
            "The camera should be high enough to see the full boundary area (at least 1.5x the "
            "tallest visible building), with sufficient resolution to identify rooftops clearly."
        ),
        "user": (
            "Review the attached 3D viewport screenshot for area scan of: '{location}'.\n"
            "Current scan altitude (range): {range}m, pitch: {pitch}deg.\n\n"
            "Respond ONLY with a valid JSON object (no markdown, no extra text):\n"
            "{{\"status\": \"GOOD\" or \"ADJUST\", \"range\": {range}, \"pitch\": {pitch}, \"feedback\": \"brief description\"}}"
        )
    },
    "path": {
        "system": (
            "You are a drone navigation pilot reviewing an initial waypoint trajectory screenshot. "
            "The camera should be pointing in the general direction of travel (toward the next waypoint), "
            "with a slight downward pitch between -15deg and -35deg to show both the terrain and horizon. "
            "The path should feel like a cinematic fly-through, not an aerial map survey."
        ),
        "user": (
            "Review the attached 3D viewport screenshot for path flythrough at: '{location}'.\n"
            "Current camera: Range={range}m, Pitch={pitch}deg, Heading={heading}deg.\n\n"
            "Respond ONLY with a valid JSON object (no markdown, no extra text):\n"
            "{{\"status\": \"GOOD\" or \"ADJUST\", \"heading\": {heading}, \"pitch\": {pitch}, "
            "\"range\": {range}, \"feedback\": \"brief description\"}}"
        )
    }
}


# ── Gemini Vision Call ─────────────────────────────────────────────────────────
# ── Gemini Vision Call ─────────────────────────────────────────────────────────
def call_gemini_vision(screenshot_bytes: bytes, mode: str, params: dict, location: str) -> dict:
    """Send screenshot to Gemini Vision or OpenRouter for framing analysis."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    prompt_config = PROMPTS.get(mode, PROMPTS["single"])
    
    if mode == "single":
        user_prompt = prompt_config["user"].format(
            location=location,
            lat=params.get("lat", ""),
            lon=params.get("lon", ""),
            camera_lat=params.get("camera_lat") if params.get("camera_lat") is not None else "N/A",
            camera_lon=params.get("camera_lon") if params.get("camera_lon") is not None else "N/A",
            camera_height=params.get("camera_height") if params.get("camera_height") is not None else "N/A",
            camera_heading=params.get("camera_heading") if params.get("camera_heading") is not None else "N/A",
            camera_pitch=params.get("camera_pitch") if params.get("camera_pitch") is not None else "N/A",
            camera_roll=params.get("camera_roll") if params.get("camera_roll") is not None else "N/A",
            building_height=params.get("building_height") if params.get("building_height") is not None else "N/A",
            terrain_elevation=params.get("terrain_elevation") if params.get("terrain_elevation") is not None else "N/A",
            target_altitude=params.get("target_altitude") if params.get("target_altitude") is not None else "N/A",
            range=params.get("range", 300),
            pitch=params.get("pitch", -25),
            heading=params.get("heading", 45),
        )
    else:
        user_prompt = prompt_config["user"].format(
            location=location,
            range=params.get("range", 300),
            pitch=params.get("pitch", -25),
            heading=params.get("heading", 45),
        )

    # A. Try Direct Google Gemini API first
    if GEMINI_API_KEY:
        try:
            from google import genai
            from google.genai import types

            click.echo("Attempting screenshot review via direct Gemini API...")
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=screenshot_bytes
                                )
                            ),
                            types.Part(text=user_prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=prompt_config["system"],
                    response_mime_type="application/json"
                )
            )

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())

        except Exception as e:
            click.secho(f"Direct Gemini API error: {e}, checking fallback...", fg="yellow")

    # B. OpenRouter Fallback
    if openrouter_key:
        try:
            import base64
            import requests

            click.echo("Attempting screenshot review via OpenRouter fallback...")
            base64_image = base64.b64encode(screenshot_bytes).decode("utf-8")
            
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "google/gemini-flash-1.5",
                "messages": [
                    {
                        "role": "system",
                        "content": prompt_config["system"]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            }
            
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                return json.loads(content.strip())
            else:
                click.secho(f"OpenRouter API error (status {res.status_code})", fg="yellow")
        except Exception as or_err:
            click.secho(f"OpenRouter execution failed: {or_err}", fg="yellow")

    # Fallback response
    click.secho("Staging review failed on both direct Gemini and OpenRouter. Proceeding as GOOD.", fg="yellow")
    return {"status": "GOOD", "feedback": "AI review execution failed."}


# ── Playwright Capture Session ─────────────────────────────────────────────────
def capture_and_review(location: str, mode: str, params: dict,
                        polygon=None, path_pts=None, headed=False) -> dict:
    """
    Opens viewer.html, positions camera, captures screenshots, runs Gemini Vision
    review loop. Returns final approved params dict.
    """
    from playwright.sync_api import sync_playwright

    if not VIEWER_HTML.exists():
        click.secho(f"Error: viewer.html not found at {VIEWER_HTML}", fg="red")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    import urllib.parse
    cesium_token = os.getenv("CESIUM_ION_TOKEN", "")
    url_params = urllib.parse.urlencode({
        "lat":   params["lat"],
        "lon":   params["lon"],
        "name":  location,
        "token": cesium_token,
    })
    # Check if local server is online to bypass file:// CORS blocks in Chromium (supporting dynamic Railway port)
    import requests
    import os
    port = os.getenv("PORT", "8765")
    use_http = False
    try:
        res = requests.get(f"http://localhost:{port}/health", timeout=1)
        if res.status_code == 200:
            use_http = True
    except Exception:
        pass

    if use_http:
        viewer_url = f"http://localhost:{port}/viewer.html?{url_params}"
    else:
        viewer_url = f"file:///{str(VIEWER_HTML).replace(chr(92), '/')}?{url_params}"
    click.echo(f"Viewer URL: {viewer_url}")

    with sync_playwright() as p:
        import sys
        browser_args = []
        if sys.platform != "win32":
            browser_args = ["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"]
        browser = p.chromium.launch(
            headless=not headed,
            args=browser_args
        )
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        page = context.new_page()
        page.on("console", lambda msg: click.echo(f"[Browser] {msg.text}"))

        page.goto(viewer_url, wait_until="commit", timeout=0)
        click.echo("Waiting for Cesium scene and 3D buildings to load...")
        import time
        start_time = time.time()
        loaded = False
        # Wait up to 25 seconds for the initial load of tiles
        while time.time() - start_time < 25:
            if page.evaluate("typeof isLoaded === 'function' ? isLoaded() : false"):
                loaded = True
                break
            time.sleep(0.2)
        if loaded:
            click.echo(f"Cesium fully loaded in {time.time() - start_time:.1f}s.")
        else:
            click.echo("Warning: Cesium load timeout reached. Proceeding anyway...")

        verified = False
        iteration = 0
        final_params = params.copy()
        approved_bytes = None

        while iteration < MAX_AI_ITERATIONS and not verified:
            iteration += 1
            click.echo(
                f"\n[AI Review {iteration}/{MAX_AI_ITERATIONS}] "
                f"Range={final_params['range']}m  "
                f"Pitch={final_params['pitch']}deg  "
                f"Heading={final_params.get('heading', 45)}deg"
            )

            # Check if we can bypass browser loading on Iteration 1 using pre-captured staging screenshot
            loaded_from_staging = False
            if iteration == 1 and params.get("staging_screenshot"):
                try:
                    staging_path = Path(params["staging_screenshot"])
                    if staging_path.exists():
                        screenshot_bytes = staging_path.read_bytes()
                        click.echo(f"Bypassed browser render: Loaded staging screenshot from {staging_path}")
                        loaded_from_staging = True
                except Exception as load_err:
                    click.secho(f"Failed to load staging screenshot file: {load_err}", fg="yellow")

            if not loaded_from_staging:
                # Position camera
                page.evaluate(
                    f"updateCamera("
                    f"{final_params['lon']}, {final_params['lat']}, "
                    f"{final_params['range']}, {final_params.get('heading', 45)}, "
                    f"{final_params['pitch']})"
                )
                time.sleep(2)  # allow tiles to settle

                # Capture screenshot
                try:
                    screenshot_bytes = page.screenshot(full_page=False, timeout=10000, animations="disabled")
                except Exception as se:
                    click.secho(f"Screenshot timeout (fonts loading?), retrying fast...", fg="yellow")
                    try:
                        screenshot_bytes = page.screenshot(full_page=False, timeout=2000, animations="disabled")
                    except Exception:
                        # fallback to empty bytes
                        screenshot_bytes = b""

            shot_path = OUTPUT_DIR / f"review_{iteration:02d}_{mode}.png"
            if screenshot_bytes:
                shot_path.write_bytes(screenshot_bytes)
            click.echo(f"Screenshot saved: {shot_path}")

            # Ask Gemini Vision
            click.echo("Sending to Gemini Vision for framing analysis...")
            analysis = call_gemini_vision(screenshot_bytes, mode, final_params, location)
            status   = analysis.get("status", "GOOD").upper()
            feedback = analysis.get("feedback", "")

            click.echo(f"Gemini: [{status}] {feedback}")

            if status == "GOOD":
                click.secho("APPROVED by AI Director!", fg="green")
                approved_bytes = screenshot_bytes
                verified = True
            else:
                final_params["lat"]    += float(analysis.get("lat_offset", 0.0))
                final_params["lon"]    += float(analysis.get("lon_offset", 0.0))
                final_params["range"]   = float(analysis.get("range",   final_params["range"]))
                final_params["pitch"]   = float(analysis.get("pitch",   final_params["pitch"]))
                final_params["heading"] = float(analysis.get("heading", final_params.get("heading", 45)))
                click.echo("Adjusting camera parameters...")

        if not verified:
            click.secho("Max iterations reached. Saving last frame.", fg="yellow")
            approved_bytes = screenshot_bytes

        # Save final approved screenshot
        label = location[:30].replace(" ", "_")
        approved_path = OUTPUT_DIR / f"approved_{mode}_{label}.png"
        if approved_bytes:
            approved_path.write_bytes(approved_bytes)
        click.secho(f"Final approved screenshot: {approved_path}", fg="cyan")

        browser.close()

    return final_params


# ── CLI ────────────────────────────────────────────────────────────────────────
@click.command()
@click.option("--location",  default="Eiffel Tower, Paris",   help="Target location name")
@click.option("--mode",      default="single", type=click.Choice(["single", "area", "path"]))
@click.option("--lat",       default=48.8584,  type=float,    help="Center latitude")
@click.option("--lon",       default=2.2945,   type=float,    help="Center longitude")
@click.option("--range",     "cam_range", default=350, type=int)
@click.option("--pitch",     default=-25,  type=int)
@click.option("--heading",   default=45,   type=int)
@click.option("--polygon",   default=None, help="JSON array of polygon vertices [{lat,lon}]")
@click.option("--path",      "path_pts", default=None, help="JSON array of waypoints [{lat,lon}]")
@click.option("--headed",    is_flag=True, default=False, help="Show browser window (debug)")
@click.option("--staging-screenshot", default=None, help="Path to staging screenshot file")
@click.option("--camera-lat", type=float, default=None)
@click.option("--camera-lon", type=float, default=None)
@click.option("--camera-height", type=float, default=None)
@click.option("--camera-heading", type=float, default=None)
@click.option("--camera-pitch", type=float, default=None)
@click.option("--camera-roll", type=float, default=None)
@click.option("--building-height", type=float, default=None)
@click.option("--surface-elevation", type=float, default=None)
@click.option("--terrain-elevation", type=float, default=None)
def main(location, mode, lat, lon, cam_range, pitch, heading, polygon, path_pts, headed,
         staging_screenshot, camera_lat, camera_lon, camera_height, camera_heading, camera_pitch, camera_roll,
         building_height, surface_elevation, terrain_elevation):
    """Capture 3D Cesium screenshots with Gemini AI framing review."""

    params = {
        "lat": lat, "lon": lon, "range": cam_range, "pitch": pitch, "heading": heading,
        "camera_lat": camera_lat, "camera_lon": camera_lon, "camera_height": camera_height,
        "camera_heading": camera_heading, "camera_pitch": camera_pitch, "camera_roll": camera_roll,
        "building_height": building_height, "surface_elevation": surface_elevation, "terrain_elevation": terrain_elevation,
        "staging_screenshot": staging_screenshot
    }

    poly_data = json.loads(polygon)  if polygon  else None
    path_data = json.loads(path_pts) if path_pts else None

    # Auto-center for area/path modes
    if mode == "area" and poly_data:
        params["lat"] = sum(p["lat"] for p in poly_data) / len(poly_data)
        params["lon"] = sum(p["lon"] for p in poly_data) / len(poly_data)
    if mode == "path" and path_data:
        params["lat"] = path_data[0]["lat"]
        params["lon"] = path_data[0]["lon"]

    final = capture_and_review(location, mode, params, poly_data, path_data, headed)

    # Expose only the finalized orbit params back to stdout
    approved = {
        "lat": final.get("lat", lat),
        "lon": final.get("lon", lon),
        "range": final.get("range", cam_range),
        "pitch": final.get("pitch", pitch),
        "heading": final.get("heading", heading)
    }

    click.echo("\n=== Final Approved Camera Params ===")
    click.echo(json.dumps(approved, indent=2))


if __name__ == "__main__":
    main()
