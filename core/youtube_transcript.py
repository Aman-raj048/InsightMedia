import re
from youtube_transcript_api import YouTubeTranscriptApi


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


def get_youtube_transcript(url: str, language: str = "english") -> str:
    video_id = extract_video_id(url)

    api = YouTubeTranscriptApi()

    if language.lower() == "hinglish":
        languages = ["hi", "en"]
    else:
        languages = ["en"]

    transcript = api.fetch(
        video_id,
        languages=languages
    )

    text = " ".join(
        snippet.text
        for snippet in transcript
    )

    if not text.strip():
        raise ValueError(
            "No transcript available for this YouTube video."
        )

    return text