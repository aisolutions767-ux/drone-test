# /// script
# dependencies = [
#   "playwright>=1.40.0",
#   "google-genai>=0.1.1",
#   "python-dotenv>=1.0.0",
#   "click>=8.1.0",
#   "opencv-python-headless>=4.8.0.0",
# ]
# ///

"""
drone_render_video.py
---------------------
Playwright script that opens viewer.html and renders the full flight video
for all 3 targeting modes. Uses AI-approved camera parameters (from
drone_capture_screenshots.py) and Playwright to:

  - SINGLE mode: Standard orbit (360deg heading sweep)
  - AREA   mode: Lawnmower grid scan over the polygon bounding box
  - PATH   mode: Linear waypoint fly-through with heading interpolation

Output: drone_video.mp4 (compiled via OpenCV)

Usage examples:
  # Single orbit (use AI-approved params):
  uv run drone_render_video.py --location "Colosseum, Rome" --lat 41.8902 --lon 12.4922 --range 250 --pitch -25 --heading 45 --frames 72

  # Area grid scan:
  uv run drone_render_video.py --location "Eiffel Area" --mode area --polygon '[{"lat":48.8582,"lon":2.2941},{"lat":48.8588,"lon":2.2948},{"lat":48.8580,"lon":2.2952}]' --range 350 --pitch -55 --frames 48

  # Waypoint path fly-through:
  uv run drone_render_video.py --location "Manhattan" --mode path --path '[{"lat":40.7484,"lon":-73.9857},{"lat":40.7505,"lon":-73.9870},{"lat":40.7520,"lon":-73.9885}]' --range 200 --pitch -20 --frames 60
"""

import os
import sys
import time
import json
import math
import tempfile
import click
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

VIEWER_HTML    = Path(__file__).parent / "viewer.html"
OUTPUT_VIDEO   = Path("drone_video.mp4")
DEFAULT_FPS    = 24


# ── Grid scan helpers ──────────────────────────────────────────────────────────
def compute_polygon_bbox(polygon: list) -> tuple:
    """Return (min_lat, max_lat, min_lon, max_lon) for a polygon vertex list."""
    lats = [p["lat"] for p in polygon]
    lons = [p["lon"] for p in polygon]
    return min(lats), max(lats), min(lons), max(lons)


def generate_grid_waypoints(polygon: list, rows: int = 6) -> list:
    """
    Generate lawnmower-pattern waypoints covering the polygon bounding box.
    Returns list of (lat, lon) tuples.
    """
    min_lat, max_lat, min_lon, max_lon = compute_polygon_bbox(polygon)
    lat_step = (max_lat - min_lat) / (rows - 1) if rows > 1 else 0
    waypoints = []
    for row in range(rows):
        lat = min_lat + row * lat_step
        if row % 2 == 0:
            waypoints.append((lat, min_lon))
            waypoints.append((lat, max_lon))
        else:
            waypoints.append((lat, max_lon))
            waypoints.append((lat, min_lon))
    return waypoints


def heading_between(lat1, lon1, lat2, lon2) -> float:
    """Calculate bearing angle (degrees) from point 1 to point 2."""
    dlon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def lerp_heading(h1, h2, t) -> float:
    """Smoothly interpolate between two headings handling 360/0 wrap."""
    diff = ((h2 - h1 + 180) % 360) - 180
    return (h1 + diff * t) % 360


# ── Frame generation ───────────────────────────────────────────────────────────
def build_frame_list(mode: str, params: dict, frames: int,
                     polygon=None, path_pts=None) -> list:
    """
    Returns ordered list of camera positions for each frame:
    Each item: {"lat": float, "lon": float, "range": float, "heading": float, "pitch": float}
    """
    base = {
        "lat":     params["lat"],
        "lon":     params["lon"],
        "range":   float(params["range"]),
        "pitch":   float(params["pitch"]),
        "heading": float(params.get("heading", 45)),
    }

    if mode == "single":
        # Orbit: full 360-degree heading sweep around a fixed center
        heading_step = 360.0 / frames
        return [
            {**base, "heading": (base["heading"] + i * heading_step) % 360}
            for i in range(frames)
        ]

    elif mode == "area" and polygon:
        # Lawnmower grid scan: overhead camera sweeping in rows
        rows = max(4, int(math.sqrt(frames)))
        grid_pts = generate_grid_waypoints(polygon, rows=rows)
        frame_list = []
        frames_per_segment = max(1, frames // len(grid_pts))

        for seg_idx in range(len(grid_pts) - 1):
            lat1, lon1 = grid_pts[seg_idx]
            lat2, lon2 = grid_pts[seg_idx + 1]
            bearing = heading_between(lat1, lon1, lat2, lon2)

            for f in range(frames_per_segment):
                t = f / frames_per_segment
                frame_list.append({
                    "lat":     lat1 + (lat2 - lat1) * t,
                    "lon":     lon1 + (lon2 - lon1) * t,
                    "range":   base["range"],
                    "pitch":   max(-90, min(-45, base["pitch"])),  # overhead
                    "heading": bearing,
                })

        # Pad remaining frames at final position
        while len(frame_list) < frames:
            frame_list.append(frame_list[-1])

        return frame_list[:frames]

    elif mode == "path" and path_pts:
        # Fly-through: move along waypoints with smooth heading interpolation
        frame_list = []
        n_segs = len(path_pts) - 1
        frames_per_seg = max(1, frames // n_segs)

        for seg_idx in range(n_segs):
            p1 = path_pts[seg_idx]
            p2 = path_pts[seg_idx + 1]
            bearing = heading_between(p1["lat"], p1["lon"], p2["lat"], p2["lon"])

            # Look ahead to next segment for smooth heading transition
            if seg_idx + 2 < len(path_pts):
                p3 = path_pts[seg_idx + 2]
                next_bearing = heading_between(p2["lat"], p2["lon"], p3["lat"], p3["lon"])
            else:
                next_bearing = bearing

            for f in range(frames_per_seg):
                t = f / frames_per_seg
                smooth_heading = lerp_heading(bearing, next_bearing, t)
                frame_list.append({
                    "lat":     p1["lat"] + (p2["lat"] - p1["lat"]) * t,
                    "lon":     p1["lon"] + (p2["lon"] - p1["lon"]) * t,
                    "range":   base["range"],
                    "pitch":   base["pitch"],
                    "heading": smooth_heading,
                })

        while len(frame_list) < frames:
            frame_list.append(frame_list[-1])

        return frame_list[:frames]

    else:
        # Fallback: static camera
        return [{**base} for _ in range(frames)]


# ── Playwright Render Loop ─────────────────────────────────────────────────────
def render_video(location: str, mode: str, params: dict, frames: int,
                 polygon=None, path_pts=None, fps: int = DEFAULT_FPS, headed: bool = False):
    """Render all frames via Playwright, compile to MP4 via OpenCV."""
    from playwright.sync_api import sync_playwright
    import cv2

    if not VIEWER_HTML.exists():
        click.secho(f"Error: viewer.html not found at {VIEWER_HTML}", fg="red")
        sys.exit(1)

    import urllib.parse
    cesium_token = os.getenv("CESIUM_ION_TOKEN", "")
    url_params = urllib.parse.urlencode({
        "lat":   params["lat"],
        "lon":   params["lon"],
        "name":  location,
        "token": cesium_token,
    })
    # Check if local server is online to bypass file:// CORS blocks in Chromium
    import requests
    use_http = False
    try:
        res = requests.get("http://localhost:8765/health", timeout=1)
        if res.status_code == 200:
            use_http = True
    except Exception:
        pass

    if use_http:
        viewer_url = f"http://localhost:8765/viewer.html?{url_params}"
    else:
        viewer_url = f"file:///{str(VIEWER_HTML).replace(chr(92), '/')}?{url_params}"

    click.echo(f"\nMode: {mode.upper()} | Frames: {frames} | FPS: {fps}")
    click.echo(f"Viewer: {viewer_url}")

    frame_positions = build_frame_list(mode, params, frames, polygon, path_pts)
    click.echo(f"Generated {len(frame_positions)} frame positions.")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not headed,
            args=["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"]
        )
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        page = context.new_page()
        page.on("console", lambda msg: click.echo(f"[Browser] {msg.text}") if "error" in msg.text.lower() else None)

        page.goto(viewer_url, wait_until="commit", timeout=0)
        init_wait = 20 if mode == "path" else 5
        click.echo(f"Waiting for Cesium to initialize ({init_wait}s)...")
        time.sleep(init_wait)

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            click.echo(f"\nRendering {frames} frames...")

            for f_idx, cam in enumerate(frame_positions):
                # Move camera to this frame's position
                page.evaluate(
                    f"updateCamera({cam['lon']}, {cam['lat']}, "
                    f"{cam['range']}, {cam['heading']}, {cam['pitch']})"
                )
                time.sleep(0.12)  # allow WebGL to settle

                frame_path = frames_dir / f"frame_{f_idx:05d}.png"
                page.screenshot(path=str(frame_path), timeout=30000)

                if (f_idx + 1) % 10 == 0 or (f_idx + 1) == frames:
                    pct = (f_idx + 1) / frames * 100
                    click.echo(f"  Rendered {f_idx + 1}/{frames} frames ({pct:.0f}%)")

            # Compile frames → MP4
            click.echo("\nCompiling frames into drone_video.mp4...")
            frame_files = sorted(frames_dir.glob("frame_*.png"))

            if not frame_files:
                click.secho("Error: No frames were rendered.", fg="red")
                browser.close()
                return

            first = cv2.imread(str(frame_files[0]))
            h, w, _ = first.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (w, h))

            for f_path in frame_files:
                writer.write(cv2.imread(str(f_path)))

            writer.release()

        browser.close()

    if OUTPUT_VIDEO.exists():
        size_mb = OUTPUT_VIDEO.stat().st_size / (1024 * 1024)
        click.secho(f"\nVideo ready: {OUTPUT_VIDEO.absolute()} ({size_mb:.1f} MB)", fg="green")
    else:
        click.secho("Error: Video file was not created.", fg="red")


# ── CLI ────────────────────────────────────────────────────────────────────────
@click.command()
@click.option("--location",  default="Eiffel Tower, Paris",   help="Target location name")
@click.option("--mode",      default="single", type=click.Choice(["single", "area", "path"]))
@click.option("--lat",       default=48.8584,  type=float)
@click.option("--lon",       default=2.2945,   type=float)
@click.option("--range",     "cam_range", default=350, type=int, help="Camera focal distance (m)")
@click.option("--pitch",     default=-25, type=int,             help="Camera pitch degrees")
@click.option("--heading",   default=45,  type=int,             help="Starting heading degrees")
@click.option("--frames",    default=72,  type=int,             help="Total frames to render")
@click.option("--fps",       default=24,  type=int,             help="Output video FPS")
@click.option("--polygon",   default=None, help="JSON polygon vertices [{lat,lon}]")
@click.option("--path",      "path_pts", default=None, help="JSON path waypoints [{lat,lon}]")
@click.option("--headed",    is_flag=True, default=False, help="Show browser (debug)")
def main(location, mode, lat, lon, cam_range, pitch, heading, frames, fps, polygon, path_pts, headed):
    """Render 3D drone flight video via Playwright + OpenCV (all 3 modes supported)."""

    params = {"lat": lat, "lon": lon, "range": cam_range, "pitch": pitch, "heading": heading}

    poly_data = json.loads(polygon)  if polygon  else None
    path_data = json.loads(path_pts) if path_pts else None

    # Auto-center
    if mode == "area" and poly_data:
        params["lat"] = sum(p["lat"] for p in poly_data) / len(poly_data)
        params["lon"] = sum(p["lon"] for p in poly_data) / len(poly_data)
    if mode == "path" and path_data:
        params["lat"] = path_data[0]["lat"]
        params["lon"] = path_data[0]["lon"]

    render_video(location, mode, params, frames, poly_data, path_data, fps, headed)


if __name__ == "__main__":
    main()
