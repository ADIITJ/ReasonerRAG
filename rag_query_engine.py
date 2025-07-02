import pickle, faiss, numpy as np, networkx as nx, requests
from sentence_transformers import SentenceTransformer
from pathlib import Path

# --- Config ---
EMB_MODEL  = "all-MiniLM-L6-v2"
LLM_MODEL  = "mistral"
OLLAMA_URL = "http://localhost:11434/api/generate"
TOP_K      = 5
GRAPH_HOPS = 1
# ---------------

# --- Load ---
data = Path("data")
chunks = pickle.load(open(data/"chunks.pkl","rb"))
sums    = pickle.load(open(data/"community_summaries.pkl","rb"))
part    = pickle.load(open(data/"community_partition.pkl","rb"))
G       = pickle.load(open(data/"chunk_graph.gpickle","rb"))
idx     = faiss.read_index(str(data/"index.faiss"))
E       = np.load(data/"embeddings.npy")
embedder= SentenceTransformer(EMB_MODEL)

# --- LLM Call ---
def call_llm(p):
    try:
        r = requests.post(OLLAMA_URL,
                          json={"model":LLM_MODEL,"prompt":p,"stream":False},
                          timeout=60)
        return r.json().get("response","") if r.status_code==200 else f"❌{r.status_code}"
    except Exception as e:
        return f"❌{e}"

# --- Retrieve & Expand ---
def retrieve(query):
    v = embedder.encode([query])
    _,ids = idx.search(v, TOP_K)
    raw = [chunks[i] for i in ids[0]]
    # graph 1‑hop
    ex=set()
    for u in ids[0]:
        for nbr,_ in nx.single_source_shortest_path_length(G,u,cutoff=GRAPH_HOPS).items():
            if nbr not in ids[0]:
                ex.add(nbr)
    exp = [chunks[i] for i in ex]
    return raw, ids[0].tolist(), exp, list(ex)

# --- Hierarchical Context ---
def get_context(q):
    raw, ids, exp, ex_ids = retrieve(q)
    comms = {part[i] for i in ids+ex_ids}
    comm_texts = [sums[f"community_{c}"] for c in comms if f"community_{c}" in sums]
    layer0 = [v for k,v in sums.items() if k.startswith("layer0_")]
    layer1 = [v for k,v in sums.items() if k.startswith("layer1_")]
    return raw, exp, comm_texts, layer0, layer1

# --- Prompt ---
def build_prompt(q, raw, exp, coms, l0, l1):
    sections = [
      f"Q: {q}",
      "1) Raw Chunks:\n" + "\n".join(f"- {c}" for c in raw),
      "2) Expanded Chunks:\n" + "\n".join(f"- {c}" for c in exp),
      "3) Community Summaries:\n" + "\n".join(f"- {c}" for c in coms),
      "4) Layer0 Clusters:\n" + "\n".join(f"- {c}" for c in l0),
      "5) Layer1 Clusters:\n" + "\n".join(f"- {c}" for c in l1),
      "Answer step‑by‑step, citing from each section."
    ]
    return "\n\n".join(sections)

# --- Query ---
def query_reasoner(q):
    raw,exp,coms,l0,l1 = get_context(q)
    prompt = build_prompt(q,raw,exp,coms,l0,l1)
    return call_llm(prompt)

if __name__=="__main__":
    while True:
        q=input("Ask (or 'exit'): ")
        if q.lower() in ("exit","quit"): break
        print(query_reasoner(q))
