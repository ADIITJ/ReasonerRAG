# ReasonerRAG

**ReasonerRAG** is a Hierarchical Graph-based Retrieval-Augmented Generation (RAG) system that leverages advanced graph algorithms, community detection, and large language models (LLMs) to provide step-by-step, context-rich answers to user queries. It features a Streamlit web interface for interactive exploration, graph visualization, and hierarchical context navigation.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [Command-Line Interface](#command-line-interface)
- [Data Artifacts](#data-artifacts)
- [Configuration](#configuration)
- [Extending & Customization](#extending--customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Hierarchical Retrieval:** Multi-level context retrieval using graph expansion, community summaries, and recursive cluster summaries.
- **Graph-Based Expansion:** Uses a chunk graph to expand retrieval beyond top-k results, capturing related information via graph hops.
- **Community Detection:** Summarizes and leverages community structure for richer, more coherent context.
- **LLM Integration:** Connects to local LLMs (e.g., Mistral via Ollama) for answer generation.
- **Interactive Web UI:** Streamlit-based interface with chat history, graph visualization, and context summaries.
- **Visual Graph Exploration:** PyVis-powered interactive graph rendering with color-coded communities and chunk highlights.
- **Efficient Caching:** Uses Streamlit’s resource caching for fast artifact loading.

---

## Architecture Overview

```
User Query
    │
    ▼
[Embedder] ──> [FAISS Index] ──> [Top-K Chunks]
    │                              │
    │                              ▼
    │                        [Graph Expansion]
    │                              │
    ▼                              ▼
[Community Detection]        [Related Chunks]
    │                              │
    └─────────────┬────────────────┘
                  ▼
         [Hierarchical Summaries]
                  │
                  ▼
             [Prompt Builder]
                  │
                  ▼
             [LLM (Ollama)]
                  │
                  ▼
             [Final Answer]
```

---

## How It Works

1. **Query Embedding:** The user’s question is embedded using a SentenceTransformer model.
2. **Top-K Retrieval:** The FAISS index retrieves the most relevant document chunks.
3. **Graph Expansion:** The chunk graph is traversed (1-hop by default) to find related chunks, expanding the context.
4. **Community Summaries:** Community detection partitions the graph; summaries for relevant communities are included.
5. **Hierarchical Summaries:** Higher-level (layered) summaries provide abstracted context.
6. **Prompt Construction:** All retrieved information is formatted into a structured prompt.
7. **LLM Reasoning:** The prompt is sent to a local LLM (e.g., Mistral via Ollama) for answer generation.
8. **Visualization:** The web UI displays the answer, supporting graph visualization and context inspection.

---

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) running locally with the desired LLM (e.g., Mistral)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [PyVis](https://pyvis.readthedocs.io/)
- [sentence-transformers](https://www.sbert.net/)
- Other dependencies: `numpy`, `networkx`, `requests`, `pickle`

### Install Python Dependencies

```bash
pip install streamlit pyvis sentence-transformers faiss-cpu numpy networkx requests
```

> **Note:** If you use a GPU, install the appropriate FAISS version.

### Prepare Data Artifacts

Place your pdf files in the `pdf/` directory and then run `rag_database_builder.py` to get:
- `chunks.pkl`
- `community_summaries.pkl`
- `community_partition.pkl`
- `chunk_graph.gpickle`
- `index.faiss`
- `embeddings.npy`

> These files are typically generated.

---

## Usage

### Web Interface

Start the Streamlit app:

```bash
streamlit run app.py
```

- Open the provided local URL in your browser.
- Enter your question in the input box and click "Run Query".
- View the answer, graph visualization, and hierarchical summaries.
- Use the sidebar to adjust the recursive layer or clear chat history.

#### UI Features

- **Chat History:** Scroll through previous queries and answers.
- **Graph View:** Interactive visualization of relevant chunks, neighbors, and communities.
- **Recursive Summaries:** Inspect summaries at different abstraction layers.
- **Legend:** Color codes for top chunks, graph neighbors, and communities.

### Command-Line Interface

You can also use the CLI for quick queries:

```bash
python rag_query_engine.py
```

- Type your question at the prompt.
- Type `exit` or `quit` to leave.

---

## Data Artifacts

All data artifacts are expected in the `data/` directory:

- `chunks.pkl`: List of text chunks.
- `community_summaries.pkl`: Dict of community and layer summaries.
- `community_partition.pkl`: Dict mapping chunk indices to community IDs.
- `chunk_graph.gpickle`: NetworkX graph of chunk relationships.
- `index.faiss`: FAISS index for fast vector search.
- `embeddings.npy`: Numpy array of chunk embeddings.

---

## Configuration

Key configuration parameters (in both `app.py` and `rag_query_engine.py`):

- `EMB_MODEL`: SentenceTransformer model name (default: `all-MiniLM-L6-v2`)
- `LLM_MODEL`: LLM name for Ollama (default: `mistral`)
- `OLLAMA_URL`: URL for Ollama API (default: `http://localhost:11434/api/generate`)
- `TOP_K`: Number of top chunks to retrieve (default: 5)
- `GRAPH_HOPS`: Number of graph hops for expansion (default: 1)
- `DATA_DIR`: Path to data directory (default: `data/`)

---

## Extending & Customization

- **Change LLM:** Update `LLM_MODEL` and ensure the model is available in Ollama.
- **Adjust Retrieval:** Modify `TOP_K` or `GRAPH_HOPS` for broader/narrower context.
- **Add Layers:** Add more abstraction layers in your summaries for deeper hierarchy.
- **UI Customization:** Edit `app.py` to change the Streamlit interface or add new features.
- **Data Pipeline:** Use your own data builder to generate the required artifacts.

---

## Troubleshooting

- **Ollama Not Running:** Ensure Ollama is running and the specified model is available.
- **Missing Data Files:** All required files must be present in the `data/` directory.
- **Dependency Issues:** Check Python and package versions; reinstall if needed.
- **Graph Not Displaying:** Ensure `pyvis` is installed and the `graph_output.html` is generated.

---

## License

This project is provided for research and educational purposes but do provide me credit while using this.

---

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [NetworkX](https://networkx.org/)
- [PyVis](https://pyvis.readthedocs.io/)
- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.com/)
- [Mistral LLM](https://mistral.ai/)

---

