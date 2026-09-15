import re
import os
import requests


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL.")


def get_youtube_transcript(
    url: str,
    language: str = "english"
) -> str:

    api_key = os.getenv("YOUTUBE_TRANSCRIPT_API_KEY")

    if not api_key:
        raise ValueError(
            "YOUTUBE_TRANSCRIPT_API_KEY is not configured."
        )

    video_id = extract_video_id(url)

    endpoint = "https://www.youtubetranscript.dev/api/v2/transcribe"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if language.lower() == "hinglish":
        language_code = "hi"
    else:
        language_code = "en"

    payload = {
        "video": video_id,
        "language": language_code,
        "source": "auto",
        "allow_asr": True,
    }

    response = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise ValueError(
            f"YouTube transcript API error: "
            f"{response.status_code} - {response.text}"
        )

    data = response.json()

    transcript_data = data.get("data", {})
    text = transcript_data.get("transcript", {}).get("text", "")

    if not text.strip():
        raise ValueError(
            "No transcript available for this YouTube video."
        )

    return text