from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_rag_assistant.config import RagConfig
from local_rag_assistant.pipeline import ask_question, build_index


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    config = RagConfig(
        data_dir=Path(args.data_dir),
        index_dir=Path(args.index_dir),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )

    if args.command == "index":
        index_path = build_index(config)
        print(f"Index created: {index_path}")
        return

    if args.command == "ask":
        result = ask_question(args.query, config)
        print(json.dumps(result, indent=2))
        return

    if args.command == "demo":
        index_path = build_index(config)
        print(f"Index created: {index_path}")
        for query in args.queries:
            print(f"\nQ: {query}")
            result = ask_question(query, config)
            print(result["answer"])
            if result["sources"]:
                print("Sources:")
                for source in result["sources"]:
                    print(f"- {source['source']} ({source['score']})")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local RAG Assistant: interview-ready local RAG."
    )
    parser.add_argument("--data-dir", default="data", help="Data folder with TXT/PDF files.")
    parser.add_argument("--index-dir", default="artifacts/index", help="Where index files are saved.")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=4)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="Build retrieval index.")

    ask = sub.add_parser("ask", help="Ask a question over indexed documents.")
    ask.add_argument("--query", required=True, help="Question to ask.")

    demo = sub.add_parser("demo", help="Run index + sample Q&A.")
    demo.add_argument(
        "--queries",
        nargs="+",
        default=[
            "What is Python?",
            "Name three types of machine learning.",
        ],
    )
    return parser
