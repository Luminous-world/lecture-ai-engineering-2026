# filename: adk_agent/glossary_agent/agent.py
# description: ADK-based glossary learning support agent.

# section: 1
# description: Define the root agent using Google ADK.
from google.adk.agents import Agent
from app import tools
from app.agent import INSTRUCTIONS

root_agent = Agent(
    model="gemini-3.5-flash-lite",
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    tools=[
        tools.search_glossary,
        tools.get_lecture_summary,
        tools.make_quiz,
    ],
)