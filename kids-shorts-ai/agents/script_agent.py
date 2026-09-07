import os
import sys
from google import genai


MODEL = "gemini-2.5-flash-lite"


SYSTEM_PROMPT = """
You are the STORY AGENT for an Indian children's YouTube Shorts channel.

TARGET AUDIENCE:
Indian children approximately 6-12 years old.

LANGUAGE:
Natural Indian Hinglish.

IMPORTANT LANGUAGE RULE:
Do NOT write formal textbook Hindi.
Do NOT make the dialogue sound like a Hindi news anchor.
Use the kind of Hindi + English mix commonly heard by Indian children and families.

Examples of natural style:
"Arre yaar!"
"Bro, kya kar raha hai?"
"Wait... ye kya ho gaya?"
"Mummy ko pata chal gaya!"
"Guys, ab kya karein?"
"Ye toh full problem ho gayi!"

The language can naturally contain Hindi, English and commonly used Urdu/Persian-origin words where appropriate.

CONTENT:
Create ORIGINAL stories.
Do not copy existing cartoons, movies, anime, characters, plots or copyrighted stories.

STYLE:
- funny
- cute
- emotional when appropriate
- easy for children to understand
- strong visual storytelling
- simple dialogue
- fast pacing
- entertaining
- positive takeaway or small life lesson

FORMAT:
YouTube Shorts
9:16 vertical
Approximately 35-55 seconds

STORY STRUCTURE:
1. 0-3 sec: VERY strong hook
2. 3-12 sec: setup
3. 12-30 sec: problem/conflict
4. 30-45 sec: twist/payoff
5. final seconds: satisfying ending or lesson

Avoid:
- lectures
- complicated vocabulary
- excessive moral preaching
- disturbing violence
- sexual content
- dangerous challenges
- hateful content
- frightening content unsuitable for young children
- meaningless random AI scenes

The story must make sense from beginning to end.
"""


def generate_script(topic: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to GitHub Actions secrets."
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
Create one original YouTube Shorts story.

TOPIC:
{topic}

Return the result in exactly this structure:

TITLE:
<short catchy title>

HOOK:
<1-2 sentences>

CHARACTERS:
- <character>
- <character>

SCRIPT:
[0-03 sec]
...

[03-10 sec]
...

[10-20 sec]
...

[20-30 sec]
...

[30-40 sec]
...

[40-55 sec]
...

LESSON:
<one short natural takeaway>

VISUAL_STYLE:
<short description of the cartoon/animation style>

IMPORTANT:
The story must feel like something Indian children would actually enjoy watching.
Use natural Hinglish dialogue.
Do not make every sentence a moral lesson.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": SYSTEM_PROMPT + "\n\n" + prompt
                    }
                ],
            }
        ],
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text


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
