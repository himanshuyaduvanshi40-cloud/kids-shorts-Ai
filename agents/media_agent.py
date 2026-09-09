import json
import os
import re
from pathlib import Path

from gradio_client import Client


OUTPUT_DIR = Path("outputs/media")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SPACE = os.environ.get(
    "IMAGE_SPACE",
    "mrfakename/Z-Image-Turbo"
)

IMAGE_API = os.environ.get(
    "IMAGE_API",
    "/generate_image"
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


def build_character_reference(story):
    characters = story.get("characters", [])

    result = []

    for character in characters:
        result.append(
            f"""
Character name: {character.get('name', '')}
Age: {character.get('age', '')}
Gender: {character.get('gender', '')}
Appearance: {character.get('appearance', '')}
Clothing: {character.get('clothing', '')}
Personality: {character.get('personality', '')}
""".strip()
        )

    return "\n\n".join(result)


def build_image_prompt(story, scene):

    style = story.get(
        "visual_style",
        "bright expressive 3D cartoon animation"
    )

    character_reference = build_character_reference(
        story
    )

    prompt = f"""
Create a high-quality children's cartoon frame.

GLOBAL VISUAL STYLE:
{style}

CHARACTER REFERENCE:
{character_reference}

CURRENT SCENE:
{scene.get('visual_description', '')}

ACTION:
{scene.get('action', '')}

EMOTION:
{scene.get('emotion', '')}

BACKGROUND:
{scene.get('background', '')}

CAMERA:
{scene.get('camera', '')}

CRITICAL CHARACTER CONSISTENCY:
The characters must look exactly consistent
with the character reference.

Keep:
- face
- hairstyle
- clothing
- colors
- body proportions
- age
- accessories

consistent.

The image must be:
- child friendly
- colorful
- expressive
- cinematic
- clean
- visually readable
- vertical 9:16

Do not add text.
Do not add subtitles.
Do not add watermark.
Do not use copyrighted characters.

Vertical 9:16 composition.
"""

    return " ".join(prompt.split())


def generate_image(prompt, output_path):

    print("   Connecting to image provider...")

    client = Client(IMAGE_SPACE)

    print("   Generating image...")

    result = client.predict(
        prompt,
        api_name=IMAGE_API
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
            "Image provider returned no result."
        )

    source = Path(result)

    if not source.exists():
        raise RuntimeError(
            f"Generated image does not exist: {source}"
        )

    source.replace(output_path)


def main():

    print("🎨 Media Agent")
    print("==========================")

    story = load_story()

    scenes = story.get("scenes", [])

    if not scenes:
        raise RuntimeError(
            "No scenes found."
        )

    print(f"Found {len(scenes)} scenes.")
    print()

    for index, scene in enumerate(scenes, start=1):

        scene_number = scene.get(
            "scene_number",
            index
        )

        output_file = (
            OUTPUT_DIR /
            f"scene_{int(scene_number):02d}.png"
        )

        print(
            f"🎨 Scene {scene_number}/{len(scenes)}"
        )

        # Avoid regenerating an existing image.
        if output_file.exists():
            print(
                f"   ⏭️ Already exists: {output_file}"
            )
            print()
            continue

        prompt = build_image_prompt(
            story,
            scene
        )

        generate_image(
            prompt,
            output_file
        )

        print(
            f"   ✅ Saved {output_file}"
        )
        print()

    print("==========================")
    print("✅ ALL SCENE IMAGES READY")
    print("==========================")


if __name__ == "__main__":
    main()
