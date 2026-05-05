# McKesson Multi-Agent Procurement Workflow

An intelligent procurement assistant built on **Microsoft Agent Framework** and **Azure AI Foundry**.
Routes user queries across a local contract knowledge base (ChromaDB) and a structured ontology agent (Microsoft Fabric),
then synthesises both outputs into a single grounded response.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│   RouterAgent   │  classifies query → splits into ontology_task / semantic_task
└────────┬────────┘
         │ parallel
    ┌────┴─────┐
    ▼           ▼
┌──────────┐  ┌─────────────────┐
│Ontology  │  │ Search Agent    │
│Agent     │  │ (local ChromaDB)│
│Foundry   │  │ fastembed +     │
│Fabric    │  │ gpt-4o grounded │
└──────────┘  └─────────────────┘
    │                │
    └────────┬───────┘
             ▼
    ┌──────────────────┐
    │ SynthesizerAgent │  merges both outputs into one unified answer
    └──────────────────┘
             │
             ▼
      Final Response
```

| Agent | Role | Backend |
|---|---|---|
| RouterAgent | Classifies query, splits into sub-tasks | gpt-4o via Foundry |
| McK-StructuredAgent | Structured data — entities, hierarchies, relationships | Azure AI Foundry / Fabric |
| SearchAgent | Unstructured data — contracts, terms, SLAs | Local ChromaDB + fastembed |
| SynthesizerAgent | Merges both outputs into one answer | gpt-4o via Foundry |

---

## Project Structure

```
mckesson-multi-agent/
├── AgentsWithRouterSynthesizer_Procurement.py  # Main workflow entry point
├── AgentsWithRouter_Procurement.py             # Router-only variant
├── local_search_agent.py                       # ChromaDB search agent
├── index_documents.py                          # Indexes contracts into ChromaDB
├── debug_chunk.py                              # Diagnostic tool for chunking issues
├── Dockerfile                                  # Container for Foundry hosted agent
├── requirements.txt                            # Python dependencies
├── agent.yaml                                  # Foundry agent definition
├── azure.yaml                                  # azd service definition
├── .gitignore
├── infra/                                      # Bicep IaC for azd provisioning
│   ├── main.bicep
│   ├── main.parameters.json
│   ├── abbreviations.json
│   └── core/
│       ├── ai/
│       ├── host/
│       ├── monitor/
│       └── search/
└── data/
    └── knowledge/
        └── contracts/                          # PDF and DOCX supplier contracts
```

> **Note:** `data/chroma_db/` is excluded from git. Run `index_documents.py` locally to build it.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ (64-bit) | [python.org](https://www.python.org/downloads/) |
| Azure CLI | 2.x | [aka.ms/installazurecliwindows](https://aka.ms/installazurecliwindows) |
| Azure Developer CLI (azd) | 1.24+ | `winget install Microsoft.Azd` |
| Git | 2.x | [git-scm.com](https://git-scm.com/) |
| Azure AI Foundry access | — | iLink Systems subscription |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/kumarkadamilink/mckesson-multi-agent.git
cd mckesson-multi-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Authenticate with Azure

```bash
az login
azd auth login
```

Use your iLink Systems / Subramanian DATA 2025-2026 account.

### 4. Create your `.env` file

Create a `.env` file in the project root (never commit this):

```env
FOUNDRY_PROJECT_ENDPOINT=https://project-mckesson-resource.services.ai.azure.com/api/projects/project-mckesson
AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-4o
```

### 5. Build the local contract index

```bash
python index_documents.py
```

This reads all PDFs and DOCX files from `data/knowledge/contracts/`, chunks and embeds them using `fastembed` (`BAAI/bge-small-en-v1.5`), and stores them in a local ChromaDB collection at `data/chroma_db/`.

> First run downloads the embedding model (~130MB, cached after that).

Expected output:
```
Found 10 files to index...
[Branch_and_Sons_Contract.pdf]
  Extracted  : 2,618 characters
  Chunks     : 7
  Supplier   : Branch and Sons
  Contract # : MCK-2024-0067
  Spend      : $18,750,000
...
Files indexed   : 10
Total chunks    : ~70
Index ready. Run local_search_agent.py to query it.
```

---

## Running the Workflow

### Full multi-agent workflow

```bash
python AgentsWithRouterSynthesizer_Procurement.py
```

The default prompt is:
```
"Tell me the top 10 suppliers by spend and show me which categories they belong to"
```

Edit the `prompt` variable in `main()` to test other queries.

### Test the search agent standalone

```bash
python local_search_agent.py
```

Runs 5 test queries against the local ChromaDB index and prints grounded answers.

### Console trace output

```
[Router raw response]:      ← RouterAgent JSON classification
[Router reasoning]:         ← why the query was split
[Ontology task]:            ← sub-question sent to McK-StructuredAgent
[Semantic task]:            ← sub-question sent to local SearchAgent
[Ontology raw output]:      ← response from Foundry agent
[Semantic raw output]:      ← response from ChromaDB + gpt-4o
=== FINAL COMBINED RESPONSE ===
```

---

## Suppliers in the Knowledge Base

| Supplier | Category | Annual Spend |
|---|---|---|
| Miller Group | Medical Supplies & Equipment | $4,250,000 |
| Branch and Sons | Pharmaceutical Distribution | $18,750,000 |
| Moore, Henderson and Bennett | IT Infrastructure & Services | $6,100,000 |
| Hardy PLC | Logistics & Warehousing | $9,400,000 |
| Garcia-Zavala | Specialty Chemicals & Lab Reagents | $3,850,000 |
| Hernandez, Cuevas and Webb | Facility Management & Maintenance | $2,750,000 |
| Vargas PLC | Personal Protective Equipment | $5,200,000 |
| Smith-Gutierrez | Marketing & Creative Services | $1,450,000 |
| Kirby and Sons | Office Supplies & Managed Print | $980,000 |
| Peterson PLC | Consulting & Professional Services | $7,800,000 |

---

## Deploying to Azure AI Foundry

### Register the agent

```bash
azd extension install azure.ai.agents
azd ai agent init
azd up
```

This provisions:
- Azure Container Registry (ACR)
- Application Insights
- Hosted agent runtime in Foundry

### Deploy updates

```bash
azd deploy mckessonMultiAgent
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'docx'` | `pip install python-docx` |
| `MemoryError` during indexing | Already fixed — chunker uses guaranteed-progress algorithm |
| Two browser login popups | `prime_search_agent(credential)` called at startup — should be one popup only |
| `azd deploy` fails — no container registry | Run `azd up` first to provision ACR before deploying |
| `agent-framework` not found | `pip install agent-framework azure-ai-projects azure-identity` |

---

## Contributing

1. Fork the repo or request collaborator access from [@kumarkadamilink](https://github.com/kumarkadamilink)
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push and open a Pull Request: `git push origin feature/your-feature`

---

## Owner

**Platform Team — McKesson / iLink Systems**
Subscription: Subramanian DATA 2025–2026
Foundry Project: `project-mckesson` (West US)
