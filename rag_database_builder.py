import os
import logging
import fitz                                  # PyMuPDF
import spacy
import networkx as nx
import community as community_louvain
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import tiktoken
import requests
import random
from pathlib import Path

# ----------------- Configuration -----------------
CONFIG = {
    "CHUNK_TOKEN_LIMIT":        240,
    "CHUNK_OVERLAP":            30,
    "LOUVAIN_RESOLUTION":       0.4,   # higher resolution → fewer communities
    "RECURSIVE_SUMMARY_LAYERS": 3,
    "LLM_MODEL":                "mistral",
    "OLLAMA_URL":               "http://localhost:11434/api/generate",
    "EMBEDDING_MODEL":          "all-MiniLM-L6-v2",
    "SPACY_MODEL":              "en_core_web_sm",
    "PDF_DIR":                  "./pdfs",
    "OUTPUT_DIR":               "data",
    "LOG_LEVEL":                logging.INFO,
    "RANDOM_SEED":              42,
}
# --------------------------------------------------

# --- Logging Setup ---
outdir = Path(CONFIG["OUTPUT_DIR"]); outdir.mkdir(exist_ok=True)
log_file = outdir/"rag_builder.log"
logging.basicConfig(
    level=CONFIG["LOG_LEVEL"],
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Reproducibility ---
random.seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])

# --- Models & Tokenizer ---
logger.info("Loading spaCy and SentenceTransformer...")
nlp       = spacy.load(CONFIG["SPACY_MODEL"])
embedder = SentenceTransformer(CONFIG["EMBEDDING_MODEL"], device="cpu")
tokenizer = tiktoken.get_encoding("cl100k_base")

# --- LLM Wrapper ---
def call_llm(prompt: str, retries=3) -> str:
    if not prompt.strip():
        return "❌ Empty prompt"
    for i in range(retries):
        try:
            r = requests.post(
                CONFIG["OLLAMA_URL"],
                json={"model":CONFIG["LLM_MODEL"],"prompt":prompt,"stream":False},
                timeout=200
            )
            if r.status_code==200:
                return r.json().get("response","").strip()
        except Exception as e:
            logger.error(f"LLM attempt {i+1} failed: {e}")
    return "❌ LLM call failed"

# --- PDF → Text ---
def extract_text_from_pdfs(pdf_dir: str) -> list[str]:
    texts=[]
    for f in os.listdir(pdf_dir):
        if f.lower().endswith(".pdf"):
            try:
                doc=fitz.open(Path(pdf_dir)/f)
                texts.append("\n".join(p.get_text() for p in doc))
            except Exception as e:
                logger.error(f"PDF {f} error: {e}")
    return texts

# --- Chunking ---
def chunk_text(text, max_tokens, overlap):
    doc, chunks, cur, toks = nlp(text), [], [], 0
    for sent in doc.sents:
        s=sent.text.strip(); l=len(tokenizer.encode(s))
        if toks+l<=max_tokens:
            cur.append(s); toks+=l
        else:
            chunks.append(" ".join(cur))
            carry=cur[-overlap:] if len(cur)>=overlap else []
            cur=carry+[s]; toks=sum(len(tokenizer.encode(x)) for x in cur)
    if cur: chunks.append(" ".join(cur))
    return chunks

# --- Build Chunk‑Graph by Shared Entities ---
def build_chunk_graph(chunks):
    # map entity→list of chunk IDs
    ent2chunks=defaultdict(set)
    chunk_entities=[]
    for i,chunk in enumerate(chunks):
        doc=nlp(chunk)
        ents={e.text for e in doc.ents if e.text.strip()}
        chunk_entities.append(ents)
        for e in ents:
            ent2chunks[e].add(i)
    G=nx.Graph()
    G.add_nodes_from(range(len(chunks)))
    # connect chunks sharing any entity
    for e,chset in ent2chunks.items():
        for u in chset:
            for v in chset:
                if u<v:
                    G.add_edge(u,v)
    return G

# --- Communities on Chunk‑Graph ---
def detect_communities(G):
    part=community_louvain.best_partition(
        G, resolution=CONFIG["LOUVAIN_RESOLUTION"],
        random_state=CONFIG["RANDOM_SEED"]
    )
    logger.info(f"Chunk graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logger.info(f"Detected {len(set(part.values()))} communities")
    return part

# --- FAISS Index ---
def build_vector_index(chunks):
    E=embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    E=E/np.linalg.norm(E,axis=1,keepdims=True)
    idx=faiss.IndexFlatL2(E.shape[1]); idx.add(E)
    return idx,E

# --- Summarization & Checkpointing ---
def validate_summary(sm, txt):
    doc_o=nlp(txt); orig={e.text.lower() for e in doc_o.ents}
    doc_s=nlp(sm); found={e.text.lower() for e in doc_s.ents}
    bad=found-orig
    if bad: logger.warning(f"Hallucinated: {bad}")
    return sm

def format_prompt(title: str, text: str) -> str:
    return f"""You are an expert summarizer for academic and technical documents.

Title: {title}

Text:
{text}

Task:
Summarize the above content in a factual and concise manner. Avoid adding new information. Preserve key named entities and concepts. Keep it around 3-5 sentences.
"""

def summarize_all_layers(chunks, part):
    ckpt = outdir / "summaries_ckpt.pkl"
    summaries, base = {}, {}

    if ckpt.exists():
        summaries = pickle.load(open(ckpt, "rb"))
        logger.info(f"Loaded {len(summaries)} summaries from checkpoint")

    # --- Community-Level Summaries ---
    comms = sorted(set(part.values()))
    for i, com in enumerate(comms, 1):
        key = f"community_{com}"
        if key in summaries:
            continue
        ids = [idx for idx, c in part.items() if c == com]
        txt = "\n".join(chunks[j] for j in ids)
        prompt = format_prompt(f"Community {com}", txt)
        raw_summary = call_llm(prompt)
        s = validate_summary(raw_summary, txt)
        summaries[key] = s
        base[key] = s
        pickle.dump(summaries, open(ckpt, "wb"))
        logger.info(f"[{i}/{len(comms)}] Saved {key}")

    # --- Recursive Summary Layers ---
    for layer in range(CONFIG["RECURSIVE_SUMMARY_LAYERS"]):
        if len(base) < 2:
            break
        items = list(base.values())
        labels = AgglomerativeClustering(
            n_clusters=min(3, len(items))
        ).fit_predict(embedder.encode(items))
        new = {}

        for lbl in set(labels):
            key = f"layer{layer}_cluster{lbl}"
            if key in summaries:
                new[key] = summaries[key]
                continue
            grp = [items[i] for i, L in enumerate(labels) if L == lbl]
            txt = "\n".join(grp)
            prompt = format_prompt(key, txt)
            raw_summary = call_llm(prompt)
            s = validate_summary(raw_summary, txt)
            summaries[key] = s
            new[key] = s
            pickle.dump(summaries, open(ckpt, "wb"))
            logger.info(f"[layer{layer} cluster{lbl}] Saved {key}")

        base = new

    return summaries


# --- Save Artifacts ---
def save_all(chunks, G, idx, E, part, summaries):
    pickle.dump(chunks,open(outdir/"chunks.pkl","wb"))
    pickle.dump(summaries,open(outdir/"community_summaries.pkl","wb"))
    pickle.dump(part,open(outdir/"community_partition.pkl","wb"))
    pickle.dump(G,open(outdir/"chunk_graph.gpickle","wb"))
    faiss.write_index(idx,str(outdir/"index.faiss"))
    np.save(outdir/"embeddings.npy",E)
    logger.info("Saved all artifacts")

# --- Main Entry ---
def build_database():
    logger.info("Starting build")
    # 1. extract
    texts=extract_text_from_pdfs(CONFIG["PDF_DIR"])
    # 2. chunk
    chunks=[] 
    for t in texts: chunks+=chunk_text(t,CONFIG["CHUNK_TOKEN_LIMIT"],CONFIG["CHUNK_OVERLAP"])
    logger.info(f"{len(chunks)} chunks generated")
    # 3. graph
    G=build_chunk_graph(chunks)
    part=detect_communities(G)
    # 4. embeddings
    idx,E=build_vector_index(chunks)
    # 5. summaries
    sums=summarize_all_layers(chunks,part)
    # 6. save
    save_all(chunks,G,idx,E,part,sums)
    logger.info("Build complete ✅")

if __name__=="__main__":
    build_database()
