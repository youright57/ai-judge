# AI Judge — Legal RAG Assistant for Court Decision Analysis

> **Free, open-source tool that helps citizens analyze court decisions for compliance with higher court rulings, judicial doctrine, and binding legal guidance.**

***

## The Problem

Every year, thousands of people lose appellate cases — not because they are wrong on the merits, but because they cannot afford a lawyer to verify whether the lower court correctly applied binding higher court doctrine. Professional legal analysis is simply out of reach for most people.

**AI Judge bridges this gap.**

***

## What It Does

AI Judge is a Retrieval-Augmented Generation (RAG) assistant that:

- **Analyzes court decisions** against a knowledge base of higher court rulings, practice reviews, and binding judicial doctrine
- **Finds violations** — identifies where the first-instance court deviated from established legal guidance
- **Supports appellate work** — accepts appellate complaint arguments and evaluates their legal strength against the knowledge base
- **Maintains conversational context** — multi-turn chat allows the user to discuss a single case across multiple messages without losing memory
- **Provides source citations** — every conclusion is linked to the specific document and passage that supports it

***

## Key Features

| Feature | Description |
|---|---|
| 🔍 Hybrid search | BM25 + vector search over the legal knowledge base |
| 🧠 BGE Reranker | Cross-encoder reranking for high-precision legal passage retrieval |
| 📄 Parent Document Retriever | Retrieves full legal context, not just matching fragments |
| 💬 Multi-turn chat | Persistent case context across the entire conversation |
| ⚖️ Appellate support | Evaluates appeal arguments alongside the court decision |
| 🗄️ Knowledge base management | Admin panel for adding, rebuilding, and backing up the document index |
| 🔑 Flexible LLM backend | Supports Gemini API and OpenRouter |

***

## Tech Stack

- **Interface:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector store:** [ChromaDB](https://www.trychroma.com/)
- **Embeddings:** [Jina Embeddings v3](https://jina.ai/) (local, no API key required)
- **Reranker:** [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3) (local)
- **LLM:** Gemini 1.5 Pro / OpenRouter (user-supplied API key)
- **Observability:** LangSmith (optional)

***

## Who This Is For

- **Citizens** preparing an appeal against a first-instance court decision
- **Legal aid organizations** providing free assistance to low-income clients
- **Researchers** studying compliance of courts with higher court doctrine
- **Journalists** investigating systemic judicial errors
- **Law students** learning how to analyze judicial reasoning

***

## Humanitarian Mission

This project is built on three principles:

1. **Judicial transparency** — making court decisions machine-readable and verifiable against binding legal doctrine
2. **Equal access to justice** — providing professional-grade legal analysis to anyone, regardless of income
3. **Civic empowerment** — enabling citizens to independently identify violations of their rights before it is too late to appeal

The tool is deliberately open-source and free. There is no monetization. The knowledge base is built from publicly available legal documents and is freely redistributable.

***

## Getting Started

### Prerequisites

- Python 3.10+
- ~4 GB RAM (for local embedding and reranker models)
- A Gemini API key (free tier available) or an OpenRouter API key

### Installation

```bash
git clone https://github.com/youright57/ai-judge.git
cd ai-judge
pip install -r requirements.txt
```

### Run

```bash
streamlit run ai_judge.py
```

On first launch, go to the **Admin** tab to build the knowledge base from your legal documents.

***

## Project Structure

```
ai-judge/
├── ai_judge.py          # Main application
├── requirements.txt     # Python dependencies
├── source_docs/         # Place your legal documents here (not included)
├── rag_db/              # Auto-generated vector database (gitignored)
└── README.md
```

***

## Roadmap

- [ ] Multi-user server deployment (FastAPI backend)
- [ ] Public web version for citizens without technical setup
- [ ] Automated ingestion of new court rulings
- [ ] Export of analysis report as PDF
- [ ] Support for multiple legal systems and jurisdictions

***

## Contributing

Contributions are welcome — especially:
- Legal document collections for the knowledge base
- Translations of the UI into other languages
- Improvements to retrieval quality

Please open an issue before submitting a pull request.

***

## License

[MIT License](LICENSE) — free to use, modify, and distribute.

***

## About the Developer

This project is built and maintained by a single independent developer with no institutional backing, no team, and no funding. It was created using AI-assisted development, which itself demonstrates how modern AI tools enable civic-tech innovation by non-specialists.

If you find this tool useful, please ⭐ star the repository — it helps others find it.

***

*Built for citizens. Powered by open source.*
