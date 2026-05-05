import asyncio
import json

from azure.ai.projects import AIProjectClient
from azure.identity import InteractiveBrowserCredential

from agent_framework import Agent, workflow
from agent_framework.foundry import FoundryChatClient


# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
PROJECT_ENDPOINT = "https://project-mckesson-resource.services.ai.azure.com/api/projects/project-mckesson"

credential = InteractiveBrowserCredential()


# -----------------------------------------------------------
# Router agent (local — just classifies intent)
# -----------------------------------------------------------
router_client = FoundryChatClient(
    project_endpoint=PROJECT_ENDPOINT,
    model="gpt-4o",
    credential=credential,
)

router = Agent(
    name="RouterAgent",
    instructions="""
    You are a routing agent for a procurement data system.
    You have two downstream agents:

    1. ONTOLOGY — use when the question is about:
       - Relationships between entities (supplier → category → contract)
       - Hierarchies (org structure, category trees)
       - "Who is connected to", "what belongs to", "how is X related to Y"
       - Entity lookups by name or attribute

    2. SEMANTIC_MODEL — use when the question is about:
       - Numbers, metrics, KPIs (spend, savings, headcount)
       - Aggregations (top 10, total, average, sum)
       - Trends over time (monthly spend, YoY comparison)
       - Rankings and comparisons

    CRITICAL: Respond with ONLY raw JSON. No markdown, no code fences, no explanation.
    Output must start with { and end with }.

    Use this exact format:
    {
        "ontology_task": "rewritten sub-question for ontology agent, or null",
        "semantic_task": "rewritten sub-question for semantic model agent, or null",
        "reasoning": "one sentence explaining the split"
    }
    """,
    client=router_client,
)


# -----------------------------------------------------------
# Helper — strip markdown fences from LLM response
# -----------------------------------------------------------
def strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


# -----------------------------------------------------------
# Helper — call a deployed Foundry agent by NAME (new Foundry API)
# -----------------------------------------------------------
def call_foundry_agent_by_name(
    project_client: AIProjectClient,
    agent_name: str,
    user_message: str,
) -> str:
    openai_client = project_client.get_openai_client()

    conversation = openai_client.conversations.create()

    response = openai_client.responses.create(
        conversation=conversation.id,
        extra_body={
            "agent_reference": {
                "name": agent_name,
                "type": "agent_reference"
            }
        },
        input=user_message,
    )

    return response.output_text


# -----------------------------------------------------------
# Workflow
# -----------------------------------------------------------
@workflow
async def procurement_router_workflow(user_prompt: str) -> str:

    # Step 1: Router classifies and splits the prompt
    router_raw = (await router.run(user_prompt)).text

    print(f"\n[Router raw response]:\n{router_raw}\n")

    cleaned = strip_json_fences(router_raw)
    tasks = json.loads(cleaned)

    ontology_task = tasks.get("ontology_task")
    semantic_task = tasks.get("semantic_task")
    reasoning     = tasks.get("reasoning", "")

    print(f"[Router reasoning]: {reasoning}")
    print(f"[Ontology task]:    {ontology_task}")
    print(f"[Semantic task]:    {semantic_task}\n")

    # Step 2: Create project client for calling deployed Foundry agents
    project_client = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=credential,
    )

    # Step 3: Run agents in parallel (only those with a task)
    results = {}
    loop = asyncio.get_event_loop()

    async def run_ontology():
        if ontology_task:
            results["ontology"] = await loop.run_in_executor(
                None,
                call_foundry_agent_by_name,
                project_client,
                "McK-StructuredAgent",      # ontology agent — unchanged
                ontology_task,
            )

    async def run_semantic():
        if semantic_task:
            results["semantic"] = await loop.run_in_executor(
                None,
                call_foundry_agent_by_name,
                project_client,
                "Mck-Agent",                # ← updated to new agent name
                semantic_task,
            )

    await asyncio.gather(run_ontology(), run_semantic())

    # Step 4: Assemble final output
    output_parts = [f"Original question: {user_prompt}\n"]

    if "ontology" in results:
        output_parts.append(
            f"=== McK-StructuredAgent (Ontology) ===\n{results['ontology']}"
        )
    if "semantic" in results:
        output_parts.append(
            f"=== Mck-Agent (Semantic Model) ===\n{results['semantic']}"
        )

    return "\n\n".join(output_parts)


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
async def main() -> None:
    prompt = "Tell me the top 10 suppliers by spend and show me which categories they belong to"
    result = await procurement_router_workflow.run(prompt)
    print(result.get_outputs()[0])


if __name__ == "__main__":
    asyncio.run(main())