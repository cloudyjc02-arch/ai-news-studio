"""
Writing Crew using OpenAI Agents SDK
Three agents with handoffs: Researcher → Writer → Editor
"""

import os
import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, Model, ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")

def get_model():
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    return OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client)

def build_writing_crew():
    model = get_model()

    editor_agent = Agent(
        name="Editor",
        model=model,
        instructions="""You are a professional news editor. Polish the article:
1. Improve the headline
2. Ensure good flow and readability
3. Fix grammar issues
4. Keep it 300-500 words
5. Add 3 bullet "Key Takeaways" at the end
Return the final article with headline at top."""
    )

    writer_agent = Agent(
        name="Writer",
        model=model,
        instructions="""You are a news writer. Write a clear news article from the research notes.
Structure: headline, lead paragraph, 3-4 body paragraphs, closing.
Neutral journalistic tone. Then hand off to Editor.""",
        handoffs=[editor_agent]
    )

    researcher_agent = Agent(
        name="Researcher",
        model=model,
        instructions="""You are a news researcher. From the provided context:
1. Extract 5 key facts
2. Note data points and quotes
3. Write a 2-3 sentence narrative summary
Format: KEY FACTS, DATA POINTS, NARRATIVE SUMMARY
Then hand off to Writer.""",
        handoffs=[writer_agent]
    )

    return researcher_agent


async def run_writing_crew(topic: str, rag_context: str, extra_notes: str = "") -> str:
    researcher = build_writing_crew()

    prompt = f"""Topic: {topic}

RAG Context:
{rag_context}

Additional Notes: {extra_notes if extra_notes else 'None'}

Research this topic, hand off to Writer, who hands off to Editor. Return final article."""

    result = await Runner.run(researcher, prompt)
    return result.final_output