# #Actionableitems , decision , questions 

import os
import json

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 2200
CHUNK_OVERLAP = 150

MODEL_NAME = "openai/gpt-oss-20b"


# --------------------------------------------------
# GROQ LLM
# --------------------------------------------------

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please check your .env file."
        )

    return ChatGroq(
        model=MODEL_NAME,
        groq_api_key=api_key,
        temperature=0.1,
        max_tokens=3000
    )


# --------------------------------------------------
# SPLIT TRANSCRIPT
# --------------------------------------------------

def split_transcript(transcript: str) -> list:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_text(transcript)


# --------------------------------------------------
# CONTENT TYPE
# --------------------------------------------------

def classify_content_type(transcript: str) -> str:

    llm = get_llm()

    prompt = f"""
Classify the following transcript into exactly one category:

MEETING
or
NOT_MEETING

MEETING:
A discussion, meeting, interview, or conversation involving
participants where tasks, decisions, or follow-up matters may be discussed.

NOT_MEETING:
A lecture, tutorial, educational video, documentary, news,
commentary, informational video, or one-person content.

Use ONLY the transcript.

Return ONLY:
MEETING
or
NOT_MEETING

Transcript:

{transcript[:1500]}
"""

    try:
        response = llm.invoke(prompt)

        result = response.content.strip().upper()

        if result.startswith("MEETING") and not result.startswith("NOT"):
            return "MEETING"

        return "NOT_MEETING"

    except Exception as e:

        print(f"Content classification error: {e}")

        return "NOT_MEETING"


# --------------------------------------------------
# EXTRACT ALL FROM ONE CHUNK
# --------------------------------------------------

def extract_from_chunk(chunk: str) -> dict:

    llm = get_llm()

    prompt = f"""
You are a strict meeting information extraction assistant.

Read the transcript carefully and extract only genuine meeting outcomes.

Return ONLY valid JSON in exactly this format:

{{
    "action_items": [],
    "key_decisions": [],
    "open_questions": []
}}

========================
ACTION ITEMS
========================

Include ONLY tasks that someone needs to do AFTER the meeting.

A task must be:
- A future action
- Something a participant is expected to do
- Explicitly stated or clearly assigned in the transcript

For each action item use:
{{
    "task": "clear description of the task",
    "owner": "person name if explicitly mentioned",
    "deadline": "deadline if explicitly mentioned"
}}

Important:
- Rewrite vague phrases using the surrounding context.
- For example, do NOT write "drop that down as an action".
  Instead describe the actual task being discussed.
- Do NOT include tasks that have already been completed.
- Do NOT include information that was only reported.
- Do NOT include suggestions unless they were accepted as an action.
- Never invent an owner or deadline.

========================
KEY DECISIONS
========================

Include ONLY decisions that were actually made by the participants.

A decision must show that the team:
- agreed
- approved
- selected
- decided
- finalized
- chose
- confirmed
- committed to a course of action

Do NOT include:
- Action items
- Tasks
- Suggestions
- Questions
- Facts
- Updates
- Past events
- Statements that something happened

For example:

"make an episode review next week"

is an ACTION ITEM, not a key decision.

========================
OPEN QUESTIONS
========================

Include ONLY genuine unresolved questions that require future clarification or follow-up.

The question must:
1. Be relevant to the meeting.
2. Be unanswered.
3. Require some future action, clarification, or decision.

Do NOT include:
- Rhetorical questions
- Polite/conversational questions
- Questions that were immediately answered
- Questions used only to encourage discussion
- Suggestions phrased as questions
- General discussion
- Questions that do not require follow-up

For example:

"Is there anything that we need to do to help you with that at all?"

should NOT be included unless the transcript clearly shows that it remained an unresolved follow-up issue.

========================
STRICT RULES
========================

- Use ONLY information from the transcript.
- Never invent names.
- Never invent tasks.
- Never invent deadlines.
- Never invent decisions.
- Never invent questions.
- Prefer fewer accurate results over many uncertain results.
- If there is doubt, leave the item out.
- Do not duplicate the same information in different categories.
- Return ONLY valid JSON.
- Do not add markdown.
- Do not add explanations.

Transcript:
{chunk}
"""

    try:

        response = llm.invoke(prompt)

        content = response.content.strip()

        # Remove markdown code block if model returns it
        if content.startswith("```"):

            content = content.replace("```json", "")
            content = content.replace("```", "")
            content = content.strip()

        data = json.loads(content)

        return {
            "action_items": data.get("action_items", []),
            "key_decisions": data.get("key_decisions", []),
            "open_questions": data.get("open_questions", [])
        }

    except Exception as e:

        print(f"Extraction error: {e}")

        return {
            "action_items": [],
            "key_decisions": [],
            "open_questions": []
        }


# --------------------------------------------------
# EXTRACT COMPLETE TRANSCRIPT
# --------------------------------------------------

def extract_all(transcript: str) -> dict:

    chunks = split_transcript(transcript)

    all_action_items = []
    all_decisions = []
    all_questions = []

    for i, chunk in enumerate(chunks):

        print(
            f"Extracting chunk {i + 1}/{len(chunks)}..."
        )

        result = extract_from_chunk(chunk)

        all_action_items.extend(
            result["action_items"]
        )

        all_decisions.extend(
            result["key_decisions"]
        )

        all_questions.extend(
            result["open_questions"]
        )

    return {
        "action_items": remove_duplicates(
            all_action_items
        ),
        "key_decisions": remove_duplicates(
            all_decisions
        ),
        "open_questions": remove_duplicates(
            all_questions
        )
    }


# --------------------------------------------------
# REMOVE DUPLICATES
# --------------------------------------------------

def remove_duplicates(items: list) -> list:

    unique = []

    for item in items:

        if isinstance(item, dict):
            item = format_action_item(item)

        clean = str(item).strip()

        if not clean:
            continue

        if clean.upper() == "NONE":
            continue

        if clean.lower() not in [
            x.lower() for x in unique
        ]:
            unique.append(clean)

    return unique


# --------------------------------------------------
# FORMAT ACTION ITEM
# --------------------------------------------------

def format_action_item(item) -> str:

    if isinstance(item, str):
        return item.strip()

    task = str(
        item.get("task", "")
    ).strip()

    owner = str(
        item.get("owner", "")
    ).strip()

    deadline = str(
        item.get("deadline", "")
    ).strip()

    if not task:
        return ""

    result = f"Task: {task}"

    if owner and owner.lower() not in [
        "none",
        "null",
        ""
    ]:
        result += f"\nOwner: {owner}"

    if deadline and deadline.lower() not in [
        "none",
        "null",
        ""
    ]:
        result += f"\nDeadline: {deadline}"

    return result


# --------------------------------------------------
# ACTION ITEMS
# --------------------------------------------------

def format_action_items(items: list) -> str:

    formatted_items = []

    for item in items:

        formatted = format_action_item(item)

        if formatted:
            formatted_items.append(formatted)

    if not formatted_items:
        return "No action items found."

    return "\n".join(
        f"{i + 1}. {item}"
        for i, item in enumerate(formatted_items)
    )


# --------------------------------------------------
# KEY DECISIONS
# --------------------------------------------------

def format_decisions(items: list) -> str:

    if not items:
        return "No key decisions found."

    return "\n".join(
        f"{i + 1}. {item}"
        for i, item in enumerate(items)
    )


# --------------------------------------------------
# OPEN QUESTIONS
# --------------------------------------------------

def format_questions(items: list) -> str:

    valid_questions = []

    for item in items:

        clean = str(item).strip()

        if "?" in clean:
            valid_questions.append(clean)

    if not valid_questions:
        return "No open questions found."

    return "\n".join(
        f"{i + 1}. {item}"
        for i, item in enumerate(valid_questions)
    )


# --------------------------------------------------
# MAIN EXTRACTION FUNCTION
# --------------------------------------------------

def extract_information(transcript: str) -> dict:

    content_type = classify_content_type(transcript)

    if content_type == "NOT_MEETING":

        return {
            "action_items": "No action items needed.",
            "key_decisions": "No key decisions found.",
            "open_questions": "No open questions found."
        }

    result = extract_all(transcript)

    return {
        "action_items": format_action_items(
            result["action_items"]
        ),

        "key_decisions": format_decisions(
            result["key_decisions"]
        ),

        "open_questions": format_questions(
            result["open_questions"]
        )
    }


# --------------------------------------------------
# COMPATIBILITY FUNCTIONS
# --------------------------------------------------
# These keep main.py compatible if needed.


def extract_action_items(transcript: str) -> str:

    return extract_information(
        transcript
    )["action_items"]


def extract_key_decisions(transcript: str) -> str:

    return extract_information(
        transcript
    )["key_decisions"]


def extract_questions(transcript: str) -> str:

    return extract_information(
        transcript
    )["open_questions"]