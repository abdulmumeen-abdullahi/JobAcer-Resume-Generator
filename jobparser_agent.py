import os
import json
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner

# -----------------------------
# Load environment variable
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Environment variable GOOGLE_API_KEY is not set. Set it and retry.")
print("GOOGLE_API_KEY loaded successfully.")

# -----------------------------
# Retry configuration
# -----------------------------
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# -----------------------------
# JSON Schema for structured output
# -----------------------------
job_parser_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "skills": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "seniority": {"type": "string"},
        "company_intent_summary": {"type": "string"},
        "preferred_tone": {"type": "string"},
        "target_sections": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["skills", "keywords", "seniority", "preferred_tone", "target_sections"]
}

# -----------------------------
# LLM model
# -----------------------------
model = Gemini(
    model="gemini-2.5-flash-lite",
    api_key=GOOGLE_API_KEY,
    retry_options=retry_config,
    schema=job_parser_schema,  # enforce structured JSON
    schema_type="json"
)

# -----------------------------
# JobParser Agent Definition
# -----------------------------
jobparser_agent = Agent(
    name="JobParserAgent",
    model=model,
    instruction="""
You are an advanced Job Parsing Agent. Analyze the job description text
(and optional company info) and return ONLY the following JSON keys exactly:
skills, keywords, seniority, company_intent_summary, preferred_tone, target_sections.
Do not add or remove keys. Ensure valid JSON output.
""",
    tools=[],  # analysis only
    output_key="parsed_job_data"
)

# -----------------------------
# Async runner function
# -----------------------------
async def run_job_parser(job_description: str):
    runner = InMemoryRunner(agent=jobparser_agent)
    # Run the agent
    response_events = await runner.run_debug(job_description)

    # Directly extract the structured JSON from agent
    raw_output = response_events[0].content.parts[0].text
    cleaned_json_str = raw_output.strip().strip("```json").strip("```").strip()
    parsed_data = json.loads(cleaned_json_str)
    
    # ✅ Ready to pass to another agent
    return parsed_data
