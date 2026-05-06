"""
server.py
---------
Thin FastAPI wrapper that exposes the procurement_router_workflow
as a Foundry-compatible HTTP server on port 8088.

The Foundry playground POSTs to /responses with:
    {"input": "user question", "stream": false}

And expects back:
    {"output_text": "agent answer", "conversation_id": "..."}

Run:
    python server.py

Requirements:
    pip install fastapi uvicorn
"""

import asyncio
import uuid
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Import the workflow and its dependencies
from AgentsWithRouterSynthesizer_Procurement import procurement_router_workflow

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="McKesson Procurement Agent",
    description="Multi-agent procurement assistant — Foundry playground compatible",
    version="1.0.0",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "running", "agent": "McKesson Procurement Assistant"}


# ─────────────────────────────────────────────
# Main responses endpoint — Foundry playground compatible
# ─────────────────────────────────────────────
@app.post("/responses")
async def responses(request: Request):
    body = await request.json()

    # Extract user input — Foundry sends either "input" or "messages"
    user_input = body.get("input") or ""
    if not user_input and "messages" in body:
        # Handle OpenAI-style messages array
        messages = body.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]
        if user_messages:
            content = user_messages[-1].get("content", "")
            user_input = content if isinstance(content, str) else str(content)

    if not user_input:
        return JSONResponse(
            status_code=400,
            content={"error": "No input provided. Send {'input': 'your question'}"}
        )

    conversation_id = body.get("conversation_id") or str(uuid.uuid4())
    logger.info(f"[{conversation_id}] Input: {user_input}")

    try:
        # Run the full multi-agent workflow
        result = await procurement_router_workflow.run(user_input)
        answer = result.get_outputs()[0]
        logger.info(f"[{conversation_id}] Answer: {answer[:100]}...")

        # Return in Foundry Responses protocol format
        return JSONResponse(content={
            "output_text": answer,
            "conversation_id": conversation_id,
            "status": "completed",
            "role": "assistant",
        })

    except Exception as e:
        logger.error(f"[{conversation_id}] Workflow error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "conversation_id": conversation_id,
                "status": "failed",
            }
        )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  McKesson Multi-Agent Procurement Server")
    print("  Listening on http://0.0.0.0:8088/responses")
    print("  Health check: http://localhost:8088/")
    print("  Open the Foundry playground to start chatting.")
    print("  Ctrl+C to stop.")
    print("=" * 55)

    uvicorn.run(app, host="0.0.0.0", port=8088)
