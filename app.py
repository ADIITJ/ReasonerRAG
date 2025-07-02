import streamlit as st
import pickle, faiss, numpy as np, networkx as nx, requests
from sentence_transformers import SentenceTransformer
from pyvis.network import Network
from pathlib import Path

# --- Config ---
EMB_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api/generate"
TOP_K = 5
GRAPH_HOPS = 1
DATA_DIR = Path("data")

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    chunks     = pickle.load(open(DATA_DIR / "chunks.pkl", "rb"))
    summaries  = pickle.load(open(DATA_DIR / "community_summaries.pkl", "rb"))
    partition  = pickle.load(open(DATA_DIR / "community_partition.pkl", "rb"))
    graph      = pickle.load(open(DATA_DIR / "chunk_graph.gpickle", "rb"))
    index      = faiss.read_index(str(DATA_DIR / "index.faiss"))
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    embedder   = SentenceTransformer(EMB_MODEL)
    return chunks, summaries, partition, graph, index, embeddings, embedder

chunks, summaries, partition, graph, index, embeddings, embedder = load_artifacts()

# --- LLM Wrapper ---
def call_llm(prompt):
    try:
        res = requests.post(OLLAMA_URL, json={"model": LLM_MODEL, "prompt": prompt, "stream": False}, timeout=400)
        return res.json().get("response", "") if res.status_code == 200 else f"LLM Error {res.status_code}"
    except Exception as e:
        return f"Exception: {e}"

# --- Retrieval ---
def retrieve_hierarchy(query, layer):
    vec = embedder.encode([query])
    _, idxs = index.search(vec, TOP_K)
    ids = idxs[0].tolist()
    retrieved = [chunks[i] for i in ids]

    # Graph-based expansion
    related = set()
    for i in ids:
        related.update(nx.single_source_shortest_path_length(graph, i, cutoff=GRAPH_HOPS).keys())
    related = list(set(related) - set(ids))

    related_vecs = embeddings[related]
    query_vec = vec[0]
    sim_scores = related_vecs @ query_vec / (np.linalg.norm(related_vecs, axis=1) * np.linalg.norm(query_vec))
    top_related = [related[i] for i in np.argsort(sim_scores)[::-1][:TOP_K]]
    expanded = [chunks[i] for i in top_related]

    # Community summaries
    comms = {partition[i] for i in ids + top_related if i in partition}
    community_summaries = [summaries.get(f"community_{c}", "") for c in list(comms)[:2]]

    # Recursive layers
    layer_summaries = [
        v for k, v in summaries.items()
        if k.startswith(f"layer{layer}_")
    ][:2]

    return retrieved, expanded, community_summaries, layer_summaries, ids, top_related

# --- Prompt Builder ---
def build_prompt(query, chunks, expanded, communities, layers):
    return f"""Q: {query}

Top Chunks:
{chr(10).join("- " + c for c in chunks[:2])}

Graph Expansion:
{chr(10).join("- " + c for c in expanded[:2])}

Community Summaries:
{chr(10).join("- " + s for s in communities[:1])}

Recursive Layer Summaries:
{chr(10).join("- " + s for s in layers[:1])}

Answer step-by-step using only the given information. Be concise and specific.
"""

# --- Graph Visualization ---
def visualize_graph(top_ids, ex_ids, partition):
    net = Network(height="500px", width="100%", bgcolor="#222", font_color="white", notebook=False)
    cluster_colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
    ]
    for n in graph.nodes():
        if n in top_ids:
            color = "#FB7E81"
        elif n in ex_ids:
            color = "#7BE141"
        elif n in partition:
            color = cluster_colors[partition[n] % len(cluster_colors)]
        else:
            color = "#999999"
        net.add_node(n, label=str(n), color=color)

    for a, b in graph.edges():
        net.add_edge(a, b)

    net.force_atlas_2based()
    html_path = "graph_output.html"
    net.save_graph(html_path)
    return html_path

# --- Streamlit UI ---
st.set_page_config(page_title="ReasonerRAG", layout="wide")
st.title("ReasonerRAG — Hierarchical Graph RAG Interface")

if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat"):
        st.session_state.history.clear()
        st.rerun()
    selected_layer = st.slider("Recursive Layer (0=base, 1+ = abstract)", 0, 2, 0)
    st.markdown("---")
    st.markdown("Legend:")
    st.markdown("- **Red**: Top Chunks")
    st.markdown("- **Green**: Graph Neighbors")
    st.markdown("- **Color**: Community")

# --- Query Area ---
query = st.text_input("Ask a question:")
if st.button("Run Query") and query:
    top_chunks, expanded_chunks, comm_summaries, layer_summaries, ids, ex_ids = retrieve_hierarchy(query, selected_layer)
    prompt = build_prompt(query, top_chunks, expanded_chunks, comm_summaries, layer_summaries)
    answer = call_llm(prompt)

    st.session_state.history.append({
        "query": query,
        "answer": answer,
        "top_ids": ids,
        "ex_ids": ex_ids,
        "partition": partition,
        "layer_summaries": layer_summaries
    })
    st.rerun()

# --- Display History ---
for i, chat in enumerate(reversed(st.session_state.history)):
    st.markdown(f"#### You: {chat['query']}")
    st.markdown(f"#### ReasonerRAG:")
    st.markdown(chat["answer"])

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("##### Graph View")
        html_path = visualize_graph(chat["top_ids"], chat["ex_ids"], chat["partition"])
        st.components.v1.html(open(html_path).read(), height=500)

    with col2:
        st.markdown("##### Recursive Summaries")
        for idx, summary in enumerate(chat["layer_summaries"]):
            st.markdown(f"**Cluster {idx} Summary:**")
            st.markdown(f"> {summary}")
