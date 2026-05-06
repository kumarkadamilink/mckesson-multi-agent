"""
local_search_agent.py
---------------------
Drop-in replacement for the Azure AI Search sub-agent call.
Queries the local ChromaDB collection using fastembed vectors,
then uses gpt-4o (via your Foundry endpoint) to synthesize
a grounded answer from the retrieved chunks.

Can be used two ways:

  1. Standalone test:
       python local_search_agent.py

  2. Imported into AgentsWithRouterSynthesizer_Procurement.py:
       from local_search_agent import search_contracts, prime_search_agent

Requirements:
    pip install chromadb onnxruntime fastembed pypdf python-docx
"""

from pathlib import Path

import chromadb
from fastembed import TextEmbedding
from azure.ai.projects import AIProjectClient
from azure.identity import InteractiveBrowserCredential

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CHROMA_DIR       = Path("data/chroma_db")
COLLECTION_NAME  = "procurement_contracts"
EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
TOP_K            = 10
PROJECT_ENDPOINT = "https://project-mckesson-resource.services.ai.azure.com/api/projects/project-mckesson"


# ─────────────────────────────────────────────
# SINGLETONS  (each loaded once, reused)
# ─────────────────────────────────────────────
_embed_model   = None
_collection    = None
_openai_client = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("[SearchAgent] Loading embedding model...")
        _embed_model = TextEmbedding(model_name=EMBED_MODEL)
    return _embed_model


def _get_collection():
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DIR}. "
                "Run index_documents.py first."
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
        print(f"[SearchAgent] Connected to '{COLLECTION_NAME}' "
              f"({_collection.count()} chunks)")
    return _collection


def _get_openai_client(credential=None):
    """
    Returns the shared OpenAI client.
    On first call, authenticates using the supplied credential.
    If no credential is supplied, creates a new InteractiveBrowserCredential.
    Subsequent calls ignore the credential argument and return the cached client.
    """
    global _openai_client
    if _openai_client is None:
        print("[SearchAgent] Authenticating with Azure Foundry (once)...")
        if credential is None:
            credential = InteractiveBrowserCredential()
        project_client = AIProjectClient(
            endpoint=PROJECT_ENDPOINT,
            credential=credential,
        )
        _openai_client = project_client.get_openai_client()
        print("[SearchAgent] Authenticated.")
    return _openai_client


def prime_search_agent(credential):
    """
    Call this at startup with the shared credential to avoid a second
    browser login when search_contracts() is first invoked.
    """
    _get_openai_client(credential=credential)


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────
# Known supplier names for auto-detection
KNOWN_SUPPLIERS = [
    "Miller Group", "Branch and Sons", "Moore Henderson and Bennett",
    "Hardy PLC", "Garcia-Zavala", "Hernandez Cuevas and Webb",
    "Vargas PLC", "Smith-Gutierrez", "Kirby and Sons", "Peterson PLC",
]

def _detect_supplier(query: str) -> str:
    """
    Auto-detect a supplier name mentioned in the query.
    Returns the supplier name if found, else None.
    """
    query_lower = query.lower()
    for supplier in KNOWN_SUPPLIERS:
        # Match on first distinctive word e.g. "Miller", "Hardy", "Peterson"
        key = supplier.split()[0].lower()
        if key in query_lower:
            return supplier
    return None


def retrieve_chunks(query: str, top_k: int = TOP_K,
                    supplier_filter: str = None) -> list:
    """
    Embed the query and retrieve the top_k most similar chunks.
    Auto-detects supplier from query if not explicitly supplied.
    When a supplier is detected, fetches ALL their chunks first
    then re-ranks by score — ensures no chunk boundary misses.
    """
    model      = _get_embed_model()
    collection = _get_collection()

    query_vec = list(model.embed([query]))[0].tolist()

    # Auto-detect supplier if not explicitly passed
    if not supplier_filter:
        supplier_filter = _detect_supplier(query)

    if supplier_filter:
        print(f"[SearchAgent] Supplier filter: {supplier_filter}")
        # Fetch all chunks for this supplier, then re-rank by vector score
        where = {"supplier": {"$eq": supplier_filter}}
        total = collection.count()
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(total, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    else:
        # No supplier detected — broad search across all contracts
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text"    : doc,
            "supplier": meta.get("supplier", "Unknown"),
            "filename": meta.get("filename", ""),
            "contract": meta.get("contract_number", ""),
            "category": meta.get("category", ""),
            "spend"   : meta.get("annual_spend", ""),
            "score"   : round(1 - dist, 4),
            "metadata": meta,
        })

    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks[:top_k]


def format_context(chunks: list) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[Chunk {i} | Supplier: {c['supplier']} | "
            f"Contract: {c['contract']} | Score: {c['score']}]\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(lines)


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
def search_contracts(query: str, top_k: int = TOP_K,
                     supplier_filter: str = None) -> str:
    """
    Retrieve relevant contract chunks and return a grounded,
    cited answer from gpt-4o. Synchronous — safe to call via
    asyncio.get_event_loop().run_in_executor().
    """
    print(f"\n[SearchAgent] Query: {query}")

    chunks = retrieve_chunks(query, top_k=top_k,
                             supplier_filter=supplier_filter)
    if not chunks:
        return "No relevant contract information found in the local index."

    print(f"[SearchAgent] Retrieved {len(chunks)} chunks "
          f"(top score: {chunks[0]['score']})")

    context       = format_context(chunks)
    openai_client = _get_openai_client()

    system_prompt = """You are a procurement contract analyst at McKesson Corporation.
You will be given retrieved excerpts from supplier contracts and a question.
Answer the question using ONLY the provided contract excerpts.
- Cite the supplier name and contract number for every fact you state.
- If the excerpts do not contain enough information, say so clearly.
- Be concise and structured. Use bullet points for multi-supplier answers.
- Do not fabricate any numbers, dates, or terms not present in the excerpts."""

    user_prompt = f"""Question: {query}

Retrieved contract excerpts:
{context}

Please answer the question based on the excerpts above."""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    print(f"[SearchAgent] Answer generated ({len(answer)} chars)")
    return answer


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
TEST_QUERIES = [
    "What are the payment terms for Miller Group?",
    "Which suppliers have Net 45 payment terms?",
    "What are the SLA penalties for Hardy PLC?",
    "Which contracts cover pharmaceutical distribution?",
    "What is the annual spend for Peterson PLC?",
]

if __name__ == "__main__":
    import asyncio

    async def run_tests():
        print("=" * 60)
        print("LOCAL SEARCH AGENT — STANDALONE TEST")
        print("=" * 60)

        loop = asyncio.get_event_loop()
        for query in TEST_QUERIES:
            print(f"\n{'─' * 60}")
            print(f"Q: {query}")
            print("─" * 60)
            answer = await loop.run_in_executor(
                None, search_contracts, query)
            print(f"A: {answer}")

        print(f"\n{'=' * 60}")
        print("All test queries complete.")

    asyncio.run(run_tests())
