import os

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


MODEL_NAME = "openai/gpt-oss-20b"


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please check your .env file."
        )

    return ChatGroq(
        model=MODEL_NAME,
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=500
    )


def split_transcript(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def clean_output(text: str) -> str:
    if not text:
        return ""

    return text.strip()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

def generate_title(transcript: str) -> str:

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a professional title generator.

Create a short and meaningful title for the transcript.

Rules:
- Use ONLY information from the transcript.
- Do not invent information.
- Maximum 8 words.
- Do not write a question.
- Return ONLY the title.
"""
        ),
        (
            "human",
            """
Transcript:

{transcript}

Generate the title.
"""
        )
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "transcript": transcript[:3000]
        })

        title = clean_output(response.content)

        if not title:
            return "Meeting Discussion"

        # Keep only first line
        title = title.splitlines()[0].strip()

        return title

    except Exception as e:
        print(f"Title generation error: {e}")
        return "Meeting Discussion"


# --------------------------------------------------
# SUMMARY FOR ONE CHUNK
# --------------------------------------------------

def summarize_chunk(chunk: str) -> str:

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a professional transcript summarization assistant.

Summarize the transcript in your own words.

STRICT RULES:
- Use ONLY information present in the transcript.
- Do not invent information.
- Do not copy sentences directly from the transcript.
- Do not write dialogue.
- Do not include greetings or conversational filler.
- Do not include rhetorical questions.
- Do not repeat information.
- Include only the most important ideas, facts,
  examples, and conclusions.
- Write 2 to 4 concise bullet points.
- Each bullet should contain one important idea.
- Keep the summary professional and easy to understand.
- Output ONLY bullet points.
"""
        ),
        (
            "human",
            """
Transcript:

{transcript}

Summary:
"""
        )
    ])

    chain = prompt | llm

    response = chain.invoke({
        "transcript": chunk
    })

    return clean_output(response.content)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

def summarize(transcript: str) -> str:

    if not transcript or not transcript.strip():
        return "No summary available."

    chunks = split_transcript(transcript)

    summaries = []

    for i, chunk in enumerate(chunks):

        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")

        try:
            result = summarize_chunk(chunk)

            if result:
                summaries.append(result)

        except Exception as e:
            print(f"Summary error in chunk {i + 1}: {e}")

    if not summaries:
        return "No summary available."

    final_points = []

    for summary in summaries:

        for line in summary.splitlines():

            line = line.strip()

            if not line:
                continue

            # Remove bullet symbols
            line = line.lstrip("-•*").strip()

            if line:
                final_points.append(line)

    if not final_points:
        return "No summary available."

    # Remove duplicate points
    unique_points = []

    for point in final_points:

        if point.lower() not in [
            p.lower() for p in unique_points
        ]:
            unique_points.append(point)

    # Maximum 8 points
    unique_points = unique_points[:8]

    return "\n".join(
        f"- {point}"
        for point in unique_points
    )
