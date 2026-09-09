import json
import os
import re
import time
from pathlib import Path

from gradio_client import Client, handle_file


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs/media")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_WIDTH = 720
IMAGE_HEIGHT = 1280

# Public Hugging Face Space.
# We keep this configurable because Spaces can change.
IMAGE_SPACE = os.environ.get(
    "IMAGE_SPACE",
    "mrfakename/Z-Image-Turbo"
)

IMAGE_API = os.environ.get(
    "IMAGE_API",
    "/generate_image"
)


# ============================================================
# HELPERS
# ============================================================

def load_story():
    filename = Path("outputs/story.json")

    if not filename.exists():
        raise FileNotFoundError(
            "outputs/story.json does not exist."
        )

    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return text.strip("_")[:60]


def create_character_reference(story):
    """
    Creates a reusable description containing all character
    consistency information.
    """

    characters = story.get("characters", [])

    if not characters:
        return "No named characters. Use the visual description."

    lines = []

    for char in characters:
        lines.append(
            f"""
Character: {char.get('name', 'Unknown')}
Age: {char.get('age', 'child')}
Gender: {char.get('gender', '')}
Appearance: {char.get('appearance', '')}
Clothing: {char.get('clothing', '')}
Personality: {char.get('personality', '')}
""".strip()
        )

    return "\n\n".join(lines)


def build_image_prompt(story, scene):
    character_reference = create_character_reference(story)

    visual_style = story.get(
        "visual_style",
        "bright expressive 3D cartoon animation"
    )

    scene_prompt = scene.get(
        "image_prompt",
        scene.get("visual_description", "")
    )

    prompt = f"""
Create a single high-quality children's cartoon frame.

VISUAL STYLE:
{visual_style}

CHARACTER CONSISTENCY:
{character_reference}

SCENE:
{scene_prompt}

SCENE ACTION:
{scene.get('action', '')}

EMOTION:
{scene.get('emotion', '')}

BACKGROUND:
{scene.get('background', '')}

CAMERA:
{scene.get('camera', '')}

IMPORTANT:
- Original fictional characters only.
- Keep character appearance identical to the descriptions.
- Keep clothing identical.
- Keep hairstyles identical.
- Child-friendly.
- Clean composition.
- Strong facial expression.
- Clear foreground and background separation.
- Designed for YouTube Shorts.
- Vertical 9:16 composition.
- No text.
- No watermark.
"""

    return " ".join(prompt.split())


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(prompt: str, output_path: Path):

    print(f"🖼 Connecting to image Space: {IMAGE_SPACE}")

    client = Client(IMAGE_SPACE)

    print("🎨 Generating image...")

    # Z-Image-Turbo spaces commonly expose a /generate_image
    # endpoint. The exact interface can change between Space
    # revisions, so we keep the endpoint configurable.

    result = client.predict(
        prompt,
        api_name=IMAGE_API
    )

    # Gradio results can be:
    # - a filepath
    # - a list/tuple
    # - a dictionary containing a path/url

    value = result

    if isinstance(value, (list, tuple)):
        value = value[0]

    if isinstance(value, dict):
        value = (
            value.get("path")
            or value.get("url")
        )

    if not value:
        raise RuntimeError(
            "Image provider returned no image."
        )

    # Copy the generated file into our repository output.
    source = Path(value)

    if not source.exists():
        raise RuntimeError(
            f"Returned image does not exist: {source}"
        )

    source.replace(output_path)

    print(f"✅ Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("🎬 Media Agent starting...")
    print()

    story = load_story()

    scenes = story.get("scenes", [])

    if not scenes:
        raise RuntimeError(
            "story.json contains no scenes."
        )

    print(f"📖 Loaded {len(scenes)} scenes.")
    print()

    # For this first test, generate ONLY ONE scene.
    # We don't want to burn time/queue capacity generating
    # every scene until the provider connection is confirmed.

    scene = scenes[0]

    scene_number = scene.get(
        "scene_number",
        1
    )

    prompt = build_image_prompt(
        story,
        scene
    )

    output_name = (
        f"scene_{int(scene_number):02d}.png"
    )

    output_path = OUTPUT_DIR / output_name

    print("────────────────────────────────")
    print(f"Scene: {scene_number}")
    print("────────────────────────────────")
    print()
    print("Prompt:")
    print(prompt)
    print()

    generate_image(
        prompt,
        output_path
    )

    print()
    print("================================")
    print("✅ MEDIA TEST SUCCESSFUL")
    print("================================")
    print(f"Image: {output_path}")


if __name__ == "__main__":
    main()
