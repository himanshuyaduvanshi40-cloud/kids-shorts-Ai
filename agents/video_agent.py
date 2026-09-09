import json
import os
from pathlib import Path

from gradio_client import Client, handle_file


OUTPUT_DIR = Path("outputs/media")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_FILE = OUTPUT_DIR / "scene_01.png"
VIDEO_FILE = OUTPUT_DIR / "scene_01.mp4"

VIDEO_SPACE = os.environ.get(
    "VIDEO_SPACE",
    "zerogpu-aoti/wan2-2-fp8da-aoti-faster"
)


def load_story():

    with open(
        "outputs/story.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_prompt(scene):

    return f"""
Animate this children's cartoon image.

Action:
{scene.get("action", "")}

Emotion:
{scene.get("emotion", "")}

Camera:
{scene.get("camera", "")}

Create smooth, natural movement.

Keep the exact character appearance,
clothing, hairstyle, proportions and colors.

Do not change the character design.

Do not add characters.

Cute polished children's animation.

Vertical portrait composition.

No text.
No subtitles.
No watermark.

Avoid:
distorted faces,
extra fingers,
extra limbs,
duplicated characters,
sudden camera movement.
"""


def generate_video():

    story = load_story()

    scenes = story.get("scenes", [])

    if not scenes:
        raise RuntimeError(
            "story.json contains no scenes."
        )

    if not IMAGE_FILE.exists():
        raise FileNotFoundError(
            IMAGE_FILE
        )

    scene = scenes[0]

    prompt = build_prompt(scene)

    print("🎬 Connecting to Wan 2.2...")
    print(f"Space: {VIDEO_SPACE}")

    client = Client(VIDEO_SPACE)

    print("📤 Uploading scene image...")
    print("⏳ Waiting for GPU...")
    print("🎞️ Generating video...")

    result = client.predict(
        handle_file(str(IMAGE_FILE)),
        prompt,
        6,
        "色调艳丽, 过曝, 静态, 细节模糊不清, "
        "字幕, 静止, 最差质量, 低质量, "
        "多余的手指, 畸形的, 杂乱背景",
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
            "No video was returned."
        )

    source = Path(result)

    if not source.exists():
        raise RuntimeError(
            f"Returned video does not exist: {source}"
        )

    source.replace(VIDEO_FILE)

    print()
    print("✅ Video generated!")
    print(f"📁 {VIDEO_FILE}")


if __name__ == "__main__":
    generate_video()
