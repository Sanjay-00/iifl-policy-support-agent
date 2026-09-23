import argparse
import json
import sys

from src.llm import Gemini, MissingApiKey
from src.pipeline import SupportAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy-aware customer support agent")
    parser.add_argument("question", nargs="?", help="customer question")
    parser.add_argument("--batch", help="JSON file with a list of questions")
    parser.add_argument("--debug", action="store_true", help="include retrieval and grounding diagnostics")
    parser.add_argument("--retriever", choices=["embedding", "bm25"], default="embedding")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page

    if args.batch:
        with open(args.batch, encoding="utf-8") as f:
            questions = json.load(f)
    elif args.question is not None:
        questions = [args.question]
    else:
        parser.error("pass a question or --batch FILE")

    try:
        agent = SupportAgent(Gemini(), retrieval_mode=args.retriever)
    except MissingApiKey as exc:
        sys.exit(str(exc))

    for question in questions:
        result = agent.answer(question)
        if not args.debug:
            result.pop("debug")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
