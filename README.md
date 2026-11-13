# Query Search using Pinecone and LangChain

Semantic search and question answering over your own PDF files. Documents are
split into chunks, embedded with Google Gemini, and stored in a Pinecone
serverless index. A question is embedded the same way, matched against the
index, and the closest chunks are passed to Gemini to write a grounded answer
with page-level sources.

```
PDFs -> chunks -> Gemini embeddings -> Pinecone
                                          |
question -> Gemini embedding -> top-k chunks -> Gemini answer + sources
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then add your keys
```

You need a [Google AI Studio](https://aistudio.google.com/apikey) key and a
[Pinecone](https://app.pinecone.io) key. Both have free tiers.

## Usage

Index a folder of PDFs (the index is created on first run):

```bash
python query_search.py ingest ./documents
```

Ask a question:

```bash
python query_search.py ask "What does the report say about revenue?"
```

List the matching chunks without generating an answer:

```bash
python query_search.py ask "revenue" --no-answer --top-k 6
```

## Configuration

| Variable | Required | Default |
| --- | --- | --- |
| `GOOGLE_API_KEY` | yes | |
| `PINECONE_API_KEY` | yes | |
| `PINECONE_INDEX` | no | `pdf-query-search` |
| `PINECONE_CLOUD` / `PINECONE_REGION` | no | `aws` / `us-east-1` |
| `EMBEDDING_MODEL` | no | `models/gemini-embedding-001` |
| `CHAT_MODEL` | no | `gemini-2.5-flash` |

The index dimension is read from the embedding model at ingest time. If you
change `EMBEDDING_MODEL` later, use a new index name.

## Security

Keys live in `.env`, which is git-ignored. Never commit it.
