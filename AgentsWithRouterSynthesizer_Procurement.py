import asyncio
import json

from azure.ai.projects import AIProjectClient
from azure.identity import InteractiveBrowserCredential

from agent_framework import Agent, workflow
from agent_framework.foundry import FoundryChatClient

from local_search_agent import search_contracts, prime_search_agent
from azure.monitor.opentelemetry import configure_azure_monitor


# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
PROJECT_ENDPOINT = "https://project-mckesson-resource.services.ai.azure.com/api/projects/project-mckesson"

# Single credential instance shared across all agents
# One browser login covers the router, synthesizer, and search agent
credential = InteractiveBrowserCredential()

# Prime the search agent with the shared credential immediately —
# prevents a second browser popup when search_contracts() is first called
prime_search_agent(credential)


# -----------------------------------------------------------
# Router agent — classifies and splits the prompt
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
       - Contract terms, payment terms, SLA details, penalties

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
# Synthesizer agent — merges both outputs into one answer
# -----------------------------------------------------------
synthesizer = Agent(
    name="SynthesizerAgent",
    instructions="""
    You are a procurement data synthesizer. You will receive:
    - The original user question
    - Output from a Semantic Model agent (spend metrics, rankings, numbers)
    - Output from an Ontology agent (category relationships, hierarchies)

    Your job is to combine both outputs into a single, unified, well-structured response
    that directly answers the user's original question.

    Rules:
    - Do NOT say "according to Agent A" or "according to Agent B" — just present the data naturally
    - Merge spend figures and category data together per supplier where possible
    - Use a clean numbered list or table format
    - Be concise — the user wants one answer, not two separate answers stitched together
    - If one source is unavailable, present what you have and note the gap clearly
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
# Helper — call a deployed Foundry agent by NAME
# (kept for ontology agent — swap out when that is also local)
# -----------------------------------------------------------
def call_foundry_agent_by_name(
    project_client: AIProjectClient,
    agent_name: str,
    user_message: str,
) -> str:
    openai_client = project_client.get_openai_client()
    conversation  = openai_client.conversations.create()
    response      = openai_client.responses.create(
        conversation=conversation.id,
        extra_body={
            "agent_reference": {
                "name": agent_name,
                "type": "agent_reference",
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
    tasks   = json.loads(cleaned)

    ontology_task = tasks.get("ontology_task")
    semantic_task = tasks.get("semantic_task")
    reasoning     = tasks.get("reasoning", "")

    print(f"[Router reasoning]: {reasoning}")
    print(f"[Ontology task]:    {ontology_task}")
    print(f"[Semantic task]:    {semantic_task}\n")

    # Step 2: Project client for ontology agent (Foundry-deployed)
    project_client = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=credential,
    )

    # Step 3: Run both agents in parallel
    results = {}
    loop    = asyncio.get_event_loop()

    async def run_ontology():
        if ontology_task:
            try:
                results["ontology"] = await loop.run_in_executor(
                    None,
                    call_foundry_agent_by_name,
                    project_client,
                    "McK-StructuredAgent",
                    ontology_task,
                )
            except Exception as e:
                print(f"[Ontology agent failed]: {e}")
                results["ontology"] = None

    async def run_semantic():
        if semantic_task:
            try:
                # Local ChromaDB search — no Foundry agent needed
                results["semantic"] = await loop.run_in_executor(
                    None,
                    search_contracts,
                    semantic_task,
                )
            except Exception as e:
                print(f"[Semantic agent failed]: {e}")
                results["semantic"] = None

    await asyncio.gather(run_ontology(), run_semantic())

    # Step 4: Synthesize
    ontology_output = results.get("ontology") or "Ontology data unavailable — Fabric connection error."
    semantic_output = results.get("semantic") or "Semantic data unavailable — Fabric connection error."

    print(f"[Ontology raw output]:\n{ontology_output}\n")
    print(f"[Semantic raw output]:\n{semantic_output}\n")

    synthesis_prompt = f"""
Original user question:
{user_prompt}

--- Semantic Model Agent output (contract terms, spend data) ---
{semantic_output}

--- Ontology Agent output (category relationships, hierarchies) ---
{ontology_output}

Please combine these into a single unified answer to the user's question.
"""

    final_response = (await synthesizer.run(synthesis_prompt)).text
    return final_response


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
async def main() -> None:
    prompt = "Tell me the top 10 suppliers by spend and show me which categories they belong to"
    result = await procurement_router_workflow.run(prompt)
    print("\n=== FINAL COMBINED RESPONSE ===\n")
    print(result.get_outputs()[0])


if __name__ == "__main__":
    asyncio.run(main())
