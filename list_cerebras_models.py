"""List Cerebras models available to your API key."""

import os

from dotenv import load_dotenv

load_dotenv()

from cerebras.cloud.sdk import Cerebras

api_key = os.environ.get("CEREBRAS_API_KEY")
if not api_key:
    raise SystemExit("CEREBRAS_API_KEY not set in .env")

client = Cerebras(api_key=api_key)

print("Available Cerebras models:\n")
for m in client.models.list().data:
    print(f"  {m.id}")
