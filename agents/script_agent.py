import os
import sys
from google import genai


MODEL = "gemini-3.5-flash-lite"


SYSTEM_PROMPT = """
You are the STORY AGENT for an Indian children's YouTube Shorts channel.

TARGET AUDIENCE:
Indian children approximately 6-12 years old.

LANGUAGE:
Natural Indian Hinglish.

IMPORTANT LANGUAGE RULE:
Do NOT write formal textbook Hindi.
Do NOT make the dialogue sound like a Hindi news anchor.

Use natural Indian conversational language containing:
- Hindi
- English
- commonly used Urdu/Persian-origin words
- casual expressions naturally used in India

Examples:
"Arre yaar!"
"Bro, kya kar raha hai?"
"Wait... ye kya ho gaya?"
"Mummy ko pata chal gaya!"
"Guys, ab kya karein?"
"Ye toh full problem ho gayi!"

CONTENT:
Create ORIGINAL stories.

Do not copy:
- existing cartoons
- movie characters
- anime characters
- existing YouTube stories
- copyrighted plots

STYLE:
- funny
- cute
- entertaining
- visually interesting
- simple
- fast paced
- emotionally engaging when appropriate
- suitable for children

FORMAT:
YouTube Shorts
Vertical 9:16
35-55 seconds

STORY STRUCTURE:
0-3 sec:
VERY strong hook.

3-12 sec:
Quick setup.

12-30 sec:
Problem/conflict.

30-45 sec:
Twist/payoff.

45-55 sec:
Ending or small natural lesson.

IMPORTANT:
The story should feel like an actual Indian kids cartoon Short.

Avoid:
- formal Hindi
- long explanations
- preaching
- disturbing violence
- sexual content
- dangerous challenges
- hateful content
- excessively scary content
- meaningless random scenes

The story must have a logical beginning, middle and ending.
"""


def generate_script(topic: str) -> str:

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from GitHub Actions secrets."
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
Create ONE original Indian kids YouTube Shorts story.

TOPIC:
{topic}

Return EXACTLY this structure:

TITLE:
<short catchy title>

HOOK:
<very strong first 1-2 lines>

CHARACTERS:
- <character 1>
- <character 2>

SCRIPT:

[0-03 sec]
<dialogue + action>

[03-10 sec]
<dialogue + action>

[10-20 sec]
<dialogue + action>

[20-30 sec]
<dialogue + action>

[30-40 sec]
<dialogue + action>

[40-55 sec]
<dialogue + action>

LESSON:
<one short natural takeaway>

VISUAL_STYLE:
<short description of the visual/cartoon style>

Make the dialogue natural Indian Hinglish.

Do NOT make the story sound like a school essay.

Do NOT copy any existing character or cartoon.
"""

    # Google's current API for new agentic applications.
    interaction = client.interactions.create(
        model=MODEL,
        input=SYSTEM_PROMPT + "\n\n" + prompt
    )

    result = interaction.output_text

    if not result:
        raise RuntimeError("Gemini returned an empty response.")

    return result


def main():

    topic = " ".join(sys.argv[1:]).strip()

    if not topic:
        topic = (
            "A mischievous Indian school kid tries to escape "
            "doing homework but gets caught in a funny way"
        )

    print("Generating story...")
    print()

    result = generate_script(topic)

    os.makedirs("outputs", exist_ok=True)

    output_file = "outputs/latest_script.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(result)

    print(result)
    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
