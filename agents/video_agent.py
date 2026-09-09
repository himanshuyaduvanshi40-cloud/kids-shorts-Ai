import json
import os
from pathlib import Path
from gradio_client import Client, handle_file
import subprocess

OUTPUT_DIR = Path("outputs/media")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_SPACE = os.environ.get(
    "VIDEO_SPACE",
    "zerogpu-aoti/wan2-2-fp8da-aoti-faster"
)


def load_story():

    story_file = Path("outputs/story.json")

    if not story_file.exists():
        raise FileNotFoundError(
            "outputs/story.json was not found."
        )

    with open(
        story_file,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def build_motion_prompt(scene):

    return f"""
Animate this children's cartoon scene.

ACTION:
{scene.get('action', '')}

EMOTION:
{scene.get('emotion', '')}

CAMERA:
{scene.get('camera', '')}

Create smooth natural movement.

Maintain the exact character design from
the input image.

Keep:
- hairstyle
- face
- clothing
- colors
- proportions
- accessories

consistent.

Keep the same environment.

Animation should be:
- cute
- expressive
- smooth
- child friendly
- visually clear

Do not introduce new characters.

Do not distort faces.

Do not create extra limbs.

Do not create duplicate characters.

Do not add text.

Do not add subtitles.

Do not add watermark.

Portrait / vertical composition.
"""

def convert_to_vertical(input_path, output_path):
    """
    Convert generated video to 1080x1920.

    A blurred enlarged copy creates the background,
    while the original video stays sharp in the center.
    """

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),

        "-filter_complex",

        (
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "gblur=sigma=35,"
            "eq=brightness=-0.15[bg];"

            "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1[fg];"

            "[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]"
        ),

        "-map",
        "[outv]",

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "20",

        "-pix_fmt",
        "yuv420p",

        "-an",

        str(output_path),
    ]

    subprocess.run(
        command,
        check=True
    )

def generate_video(
    image_path,
    prompt,
    output_path
):

    print("   Connecting to Wan 2.2...")

    client = Client(VIDEO_SPACE)

    print("   Uploading image...")
    print("   Waiting for GPU...")
    print("   Generating video...")

    result = client.predict(
        handle_file(str(image_path)),
        prompt,
        6,
        "static image, blurry, low quality, "
        "text, subtitles, watermark, "
        "deformed face, extra limbs, "
        "extra fingers, duplicate character, "
        "bad anatomy",
        3.5,
        1,
        1,
        42,
        True,
        api_name="/generate_video",
    )

    if isinstance(result, (list, tuple)):
        result = result[0]

    if isinstance(result, dict):
        result = (
            result.get("path")
            or result.get("url")
        )

    if not result:
        raise RuntimeError(
            "Video provider returned no result."
        )

    source = Path(result)

    if not source.exists():
        raise RuntimeError(
            f"Video file not found: {source}"
        )

    source.replace(output_path)


def main():

    print("🎬 Video Agent")
    print("==========================")

    story = load_story()

    scenes = story.get("scenes", [])

    if not scenes:
        raise RuntimeError(
            "No scenes found."
        )

    print(f"Found {len(scenes)} scenes.")
    print()

    for index, scene in enumerate(
        scenes,
        start=1
    ):

        scene_number = scene.get(
            "scene_number",
            index
        )

        image_path = (
            OUTPUT_DIR /
            f"scene_{int(scene_number):02d}.png"
        )

        video_path = (
            OUTPUT_DIR /
            f"scene_{int(scene_number):02d}.mp4"
        )

        print(
            f"🎞️ Scene {scene_number}/{len(scenes)}"
        )

        if not image_path.exists():
            print(
                f"   ❌ Missing image: {image_path}"
            )
            raise FileNotFoundError(
                image_path
            )

        if video_path.exists():
            print(
                f"   ⏭️ Already exists: {video_path}"
            )
            print()
            continue

        prompt = build_motion_prompt(scene)

        generate_video(
            image_path,
            prompt,
            video_path
        )

        print(
            f"   ✅ Saved {video_path}"
        )
        print()

    print("==========================")
    print("✅ ALL SCENE VIDEOS READY")
    print("==========================")


if __name__ == "__main__":
    main()
