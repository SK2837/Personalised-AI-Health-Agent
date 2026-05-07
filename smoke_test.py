"""
PHAI smoke test - verifies the LLM pipeline is wired up.

Run this once after `pip install -r requirements.txt` and after creating .env
with your GROQ_API_KEY. If you see a friendly hello message, you're good.
"""

import os
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_key_here":
        print("ERROR: GROQ_API_KEY is not set in .env")
        print("Copy .env.example to .env and paste your key from https://console.groq.com")
        sys.exit(1)

    from groq import Groq

    model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    print(f"Calling Groq with model: {model}")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are PHAI, a personalised AI health agent. "
                    "Greet the user in one short, friendly sentence and "
                    "mention you have access to gene + wearable data."
                ),
            },
            {"role": "user", "content": "Hello, please introduce yourself."},
        ],
        max_tokens=80,
    )

    reply = response.choices[0].message.content
    print("\n--- LLM reply ---")
    print(reply)
    print("-----------------\n")
    print("Smoke test passed. The LLM brain is online.")


if __name__ == "__main__":
    main()
