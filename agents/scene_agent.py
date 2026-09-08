import json
import os
from typing import List

from google import genai
from pydantic import BaseModel


MODEL = "gemini-3.5-flash-lite"


# -----------------------------
# STRUCTURED OUTPUT SCHEMA
# -----------------------------

class Character(BaseModel):
    name: str
    age: str
    gender: str
    appearance: str
    clothing: str
    personality: str


class Scene(BaseModel):
    scene_number: int
    duration_seconds: int

    narration: str
    dialogue: str

    action: str
    background: str
    camera: str
    emotion: str

    visual_description: str
    image_prompt: str


class Story(BaseModel):
    title: str
    language: str
    target_audience: str

    estimated_duration_seconds: int

    visual_style: str

    characters: List[Character]
    scenes: List[Scene]

    ending_lesson: str


# -----------------------------
# SYSTEM INSTRUCTIONS
# -----------------------------

SYSTEM_PROMPT = """
You are the SCENE DIRECTOR of an AI-generated Indian children's
YouTube Shorts production system.

Your job is to convert a finished Hinglish kids story into
a detailed scene-by-scene production plan.

TARGET:
Indian children approximately 6-12 years old.

LANGUAGE:
Natural Indian Hinglish.

Do NOT convert dialogue into formal Hindi.

IMPORTANT:
The story must remain ORIGINAL.

Never copy characters from:
- cartoons
- anime
- movies
- TV shows
- games
- existing YouTube channels

VISUAL STYLE:
Use a consistent original cartoon universe.

The same character must look the SAME throughout every scene.

Character appearance must remain consistent:
- hairstyle
- skin tone
- clothes
- accessories
- age
- body proportions

Every scene must be visually understandable without narration.

SHORT FORMAT:
Vertical YouTube Shorts
9:16
Approximately 35-55 seconds.

SCENE RULES:

Each scene should usually be 3-7 seconds.

Avoid creating too many scenes.

Prefer 6-10 scenes for a normal Short.

Each scene needs:

1. narration
2. dialogue
3. character action
4. background
5. camera direction
6. emotion
7. detailed visual description
8. image generation prompt

IMAGE PROMPT RULES:

Each image prompt must describe:
- characters
- character appearance
- clothing
- environment
- action
- emotion
- lighting
- camera
- composition

The image prompt must explicitly say:
"vertical 9:16 composition"

Do NOT include copyrighted characters.

Keep visual continuity between scenes.

The final story should have:
hook → setup → problem → escalation → payoff → ending/lesson.
"""


def load_script():

    filename = "outputs/latest_script.txt"

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Script file not found: {filename}"
        )

    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def generate_scene_plan(script: str):

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    prompt = f"""
Convert the following finished children's Shorts script into
a production-ready scene plan.

SCRIPT
----------------

{script}

----------------

Important:

Keep the original dialogue where possible.

Do not rewrite the story unnecessarily.

Create enough scenes for dynamic Shorts pacing.

Make every image_prompt self-contained.

The image generator will receive image_prompt without access
to the original script, so repeat all important character details
inside every relevant prompt.
"""

    interaction = client.interactions.create(
        model=MODEL,
        input=SYSTEM_PROMPT + "\n\n" + prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": Story.model_json_schema(),
        },
    )

    raw_json = interaction.output_text

    if not raw_json:
        raise RuntimeError(
            "Scene Director returned empty output."
        )

    try:
        story = Story.model_validate_json(raw_json)
    except Exception as error:
        print("Invalid structured output:")
        print(raw_json)
        raise RuntimeError(
            f"Could not validate scene JSON: {error}"
        )

    return story


def main():

    print("🎬 Scene Director starting...")

    script = load_script()

    print("📖 Script loaded.")

    story = generate_scene_plan(script)

    os.makedirs("outputs", exist_ok=True)

    output_file = "outputs/story.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            story.model_dump(),
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("✅ Scene plan created!")
    print(f"📁 Saved to: {output_file}")
    print()
    print(f"🎞️ Scenes: {len(story.scenes)}")
    print(f"🎭 Characters: {len(story.characters)}")


if __name__ == "__main__":
    main()
