import json
import os
from pathlib import Path

from gradio_client import Client, handle_file


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("outputs/media")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_FILE = OUTPUT_DIR / "scene_01.png"
VIDEO_FILE = OUTPUT_DIR / "scene_01.mp4"

VIDEO_SPACE = os.environ.get(
    "VIDEO_SPACE",
    "zerogpu-aoti/wan2-2-fp8da-aoti-faster"
)


# ============================================================
# LOAD STORY
# ============================================================

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


# ============================================================
# VIDEO PROMPT
# ============================================================

def build_motion_prompt(story, scene):

    action = scene.get(
        "action",
        "subtle natural movement"
    )

    emotion = scene.get(
        "emotion",
        "natural expression"
    )

    camera = scene.get(
        "camera",
        "gentle cinematic camera movement"
    )

    return f"""
Animate this children's cartoon scene.

ACTION:
{action}

EMOTION:
{emotion}

CAMERA:
{camera}

Create smooth natural animation.

Keep the character's:
- face
- hairstyle
- clothing
- body proportions
- colors

consistent with the input image.

Do not add new characters.

Do not change the scene unnecessarily.

Keep the animation cute, expressive and suitable
for Indian children.

Vertical/portrait composition.

Motion should feel like a polished children's
animated Short.

Avoid:
- distorted faces
- extra fingers
- extra limbs
- character duplication
- sudden camera jumps
- text
- subtitles
- watermark
"""


# ============================================================
# GENERATE VIDEO
# ============================================================

def generate_video(
    image_path,
    prompt,
    duration_seconds=3.5
):

    print()
    print("🎬 Connecting to Wan 2.2...")
    print(f"Space: {VIDEO_SPACE}")

    client = Client(VIDEO_SPACE)

    print("📤 Uploading image...")
    print("🎞️ Generating video...")
    print()

    # Current public Wan 2.2 Fast Space exposes:
    # image + prompt + steps + negative prompt +
    # duration + guidance + seed parameters.
    #
    # We use conservative settings for our first test.

    result = client.predict(
        handle_file(str(image_path)),
        prompt,
        6,
        "色调艳丽, 过曝, 静态, 细节模糊不清, "
        "字幕, 风格, 作品, 画作, 画面, 静止, "
        "整体发灰, 最差质量, 低质量, JPEG压缩残留, "
        "丑陋的, 残缺的, 多余的手指, "
        "画得不好的手部, 画得不好的脸部, "
        "畸形的, 毁容的, 形态畸形的肢体, "
        "手指融合, 静止不动的画面, "
        "杂乱的背景, 三条腿, 背景人很多, 倒着走",
        duration_seconds,
        1,
        1,
        42,
        True,
        api_name="/generate_video",
    )

    if not result:
        raise RuntimeError(
            "Video provider returned no result."
        )

    # Result is expected to contain the generated
    # video path as the first item.

    video_path = result

    if isinstance(video_path, (list, tuple)):
        video_path = video_path[0]

    if isinstance(video_path, dict):
        video_path = (
            video_path.get("path")
            or video_path.get("url")
        )

    if not video_path:
        raise RuntimeError(
            "Could not locate generated video."
        )

    source = Path(video_path)

    if not source.exists():
        raise RuntimeError(
            f"Generated video was not found: {source}"
        )

    source.replace(VIDEO_FILE)

    print()
    print("✅ VIDEO GENERATED")
    print(f"📁 {VIDEO_FILE}")



# ============================================================
# MAIN
# ============================================================

def main():

    print("🚀 Video Agent starting...")

    story = load_story()

    scenes = story.get("scenes", [])

    if not scenes:
        raise RuntimeError(
            "No scenes found in story.json."
        )

    if not IMAGE_FILE.exists():
        raise FileNotFoundError(
            f"Missing image: {IMAGE_FILE}"
        )

    scene = scenes[0]

    prompt = build_motion_prompt(
        story,
        scene
    )

    print()
    print("────────────────────────────")
    print("SCENE 1")
    print("────────────────────────────")
    print(prompt)

    generate_video(
        IMAGE_FILE,
        prompt,
        duration_seconds=3.5
    )

    print()
    print("================================")
    print("🎉 VIDEO TEST SUCCESSFUL")
    print("================================")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
