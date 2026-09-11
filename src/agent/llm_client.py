import os
import time

from dotenv import load_dotenv
from groq import Groq
from groq import RateLimitError


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)


def generate_reply(
    prompt: str,
    max_completion_tokens: int = 200
) -> str:

    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=max_completion_tokens,
                temperature=0.2,
                reasoning_effort="low",
            )

            return (
                response.choices[0].message.content or ""
            ).strip()

        except RateLimitError:
            wait_time = 2 ** attempt

            print(
                f"Rate limit reached. "
                f"Waiting {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Groq rate limit persisted after multiple retries."
    )