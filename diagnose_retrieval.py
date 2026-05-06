import chromadb
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="data/chroma_db")
col = client.get_collection("procurement_contracts")

queries = [
    "What are the payment terms for Miller Group?",
    "Miller Group contract payment Net 30",
    "SLA penalties Hardy PLC",
    "annual spend Peterson PLC",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print("="*60)
    vec = list(model.embed([query]))[0].tolist()
    results = col.query(
        query_embeddings=[vec],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = round(1 - dist, 4)
        supplier = meta.get("supplier", "unknown")
        print(f"\nSupplier: {supplier} | Score: {score}")
        print(doc[:400])
        print("---")
