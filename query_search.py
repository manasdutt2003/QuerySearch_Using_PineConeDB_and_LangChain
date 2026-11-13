"""Semantic search and question answering over PDFs.

Pipeline: PDF files -> text chunks -> Gemini embeddings -> Pinecone index.
Queries are embedded the same way, matched against the index, and the best
chunks are handed to Gemini to write a grounded answer.

Usage:
    python query_search.py ingest ./documents
    python query_search.py ask "What does the report say about revenue?"
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv

DEFAULT_INDEX = "pdf-query-search"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"

ANSWER_PROMPT = """Answer the question using only the context below.
If the context does not contain the answer, say that you could not find it.

Context:
{context}

Question: {question}
"""


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"Missing {name}. Copy .env.example to .env and fill it in.")
    return value


def build_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        google_api_key=require_env("GOOGLE_API_KEY"),
    )


def load_chunks(folder: str, chunk_size: int, chunk_overlap: int):
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = PyPDFDirectoryLoader(folder).load()
    if not documents:
        sys.exit(f"No PDF pages found in {folder!r}.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def ensure_index(client, name: str, dimension: int) -> None:
    """Create the serverless index on first use and wait until it is ready."""
    from pinecone import ServerlessSpec

    if name in client.list_indexes().names():
        return
    client.create_index(
        name=name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )
    while not client.describe_index(name).status["ready"]:
        time.sleep(1)


def get_store(embeddings, index_name: str):
    from langchain_pinecone import PineconeVectorStore

    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=require_env("PINECONE_API_KEY"),
    )


def ingest(args: argparse.Namespace) -> None:
    from pinecone import Pinecone

    chunks = load_chunks(args.folder, args.chunk_size, args.chunk_overlap)
    embeddings = build_embeddings()

    # Ask the model for its vector size instead of hard-coding it, so that
    # switching EMBEDDING_MODEL does not silently create a mismatched index.
    dimension = len(embeddings.embed_query("dimension probe"))
    client = Pinecone(api_key=require_env("PINECONE_API_KEY"))
    ensure_index(client, args.index, dimension)

    get_store(embeddings, args.index).add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks into '{args.index}'.")


def ask(args: argparse.Namespace) -> None:
    from langchain_google_genai import ChatGoogleGenerativeAI

    store = get_store(build_embeddings(), args.index)
    matches = store.similarity_search_with_score(args.question, k=args.top_k)
    if not matches:
        sys.exit("The index returned no matches. Run the ingest command first.")

    if not args.no_answer:
        context = "\n\n".join(doc.page_content for doc, _ in matches)
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL),
            google_api_key=require_env("GOOGLE_API_KEY"),
            temperature=0,
        )
        reply = llm.invoke(
            ANSWER_PROMPT.format(context=context, question=args.question)
        )
        print(f"Answer:\n{reply.content}\n")

    print("Sources:")
    for doc, score in matches:
        source = os.path.basename(str(doc.metadata.get("source", "unknown")))
        page = doc.metadata.get("page")
        location = f"{source}, page {page + 1}" if isinstance(page, int) else source
        preview = " ".join(doc.page_content.split())[:160]
        print(f"  [{score:.3f}] {location}: {preview}...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--index",
        default=os.getenv("PINECONE_INDEX", DEFAULT_INDEX),
        help="Pinecone index name (default: %(default)s)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = commands.add_parser("ingest", help="Embed a folder of PDFs")
    ingest_cmd.add_argument("folder", help="Folder containing PDF files")
    ingest_cmd.add_argument("--chunk-size", type=int, default=800)
    ingest_cmd.add_argument("--chunk-overlap", type=int, default=100)
    ingest_cmd.set_defaults(func=ingest)

    ask_cmd = commands.add_parser("ask", help="Query the index")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--top-k", type=int, default=4)
    ask_cmd.add_argument(
        "--no-answer",
        action="store_true",
        help="Only list matching chunks, skip the generated answer",
    )
    ask_cmd.set_defaults(func=ask)
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
