import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_tokens=300
    )


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)

    retriever = get_retriever(
        vector_store,
        k=6
    )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an AI meeting assistant.

Answer the user's question ONLY using the provided
meeting transcript context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the answer is not present in the context, say:
  "I could not find this information in the meeting transcript."
- Keep the answer concise and clear.
- Answer comparison questions by combining relevant information from the context.
- If the exact wording of the question is not present, use related information from the context to answer.

Meeting transcript context:

{context}
"""
        ),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain(transcript: str):
    vector_store = load_vector_store(transcript)

    retriever = get_retriever(
        vector_store,
        k=4
    )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an AI meeting assistant.

Answer the user's question ONLY using the provided
meeting transcript context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the answer is not present in the context, say:
  "I could not find this information in the meeting transcript."
- Keep the answer concise and clear.

Meeting transcript context:

{context}
"""
        ),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str):
    try:
        answer = rag_chain.invoke(question)
        return answer

    except Exception as e:
        error_message = str(e)

        if "429" in error_message or "rate limit" in error_message.lower():
            return "Groq API rate limit exceeded. Please wait and try again."

        return "Sorry, I could not process your question right now."