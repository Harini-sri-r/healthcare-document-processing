import os
import json
from pathlib import Path 

from huggingface_hub import InferenceClient


MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

IMAGE_PATH = Path(
    "data/raw/lab_reports/"
    "AHD-0425-PA-0007561_JITENDRA TRIVEDI DS_28-04-2025_1019-21_AM.pdf_page_9.png"
)

PROMPT = """
Analyze this medical document image and extract the information that is
actually visible in the document.

Return ONLY valid JSON.

Rules:
- Dynamically determine the JSON structure from the document.
- Do not use a fixed schema.
- Preserve the original visible labels, values and units.
- Preserve reference ranges separately when visible.
- Do not guess or invent information.
- If information is unreadable, use null or "[UNCLEAR]".
- Keep values associated with their correct row/label.
- Do not diagnose diseases.
- Do not provide treatment recommendations.
- Do not add explanations outside the JSON.
"""


def main():
    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError("HF_TOKEN is not configured.")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    client = InferenceClient(
        provider="featherless-ai",
        api_key=token
    )

    print("Sending image to Qwen...")
    print(f"Image: {IMAGE_PATH}")

    with open(IMAGE_PATH, "rb") as image_file:
        image_bytes = image_file.read()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                + __import__("base64")
                                .b64encode(image_bytes)
                                .decode("utf-8")
                            )
                        },
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ],
        max_tokens=1500,
    )

    output = response.choices[0].message.content

    print("\n========== QWEN OUTPUT ==========\n")
    print(output)

    print("\n========== OUTPUT TYPE ==========\n")
    print(type(output).__name__)


if __name__ == "__main__":
    main()