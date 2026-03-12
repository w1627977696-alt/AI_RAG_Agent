"""Entry point for the AI RAG Agent interactive CLI."""

import sys

from src.config import Config
from src.rag_agent import RAGAgent


def main() -> None:
    """Run an interactive RAG Agent session."""
    print("=" * 60)
    print("  AI RAG Agent")
    print("=" * 60)

    try:
        config = Config()
        config.validate()
    except ValueError as exc:
        print(f"\nConfiguration error: {exc}")
        sys.exit(1)

    agent = RAGAgent(config=config)

    print("\nCommands:")
    print("  add-file <path>       Load a file into the knowledge base")
    print("  add-dir  <path>       Load a directory into the knowledge base")
    print("  add-text              Enter text directly (end with '---')")
    print("  clear                 Clear the knowledge base")
    print("  quit                  Exit")
    print("\nAnything else is treated as a question.\n")

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        if line.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        elif line.lower().startswith("add-file "):
            path = line[len("add-file "):].strip()
            try:
                n = agent.add_file(path)
                print(f"Added {n} chunk(s) from '{path}'.")
            except FileNotFoundError as exc:
                print(f"Error: {exc}")

        elif line.lower().startswith("add-dir "):
            path = line[len("add-dir "):].strip()
            try:
                n = agent.add_directory(path)
                print(f"Added {n} chunk(s) from directory '{path}'.")
            except NotADirectoryError as exc:
                print(f"Error: {exc}")

        elif line.lower() == "add-text":
            print("Enter text (finish with a line containing only '---'):")
            lines = []
            while True:
                try:
                    tline = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if tline.strip() == "---":
                    break
                lines.append(tline)
            text = "\n".join(lines)
            if text.strip():
                n = agent.add_text(text)
                print(f"Added {n} chunk(s) from text.")

        elif line.lower() == "clear":
            agent.clear_knowledge_base()
            print("Knowledge base cleared.")

        else:
            try:
                answer = agent.query(line)
                print(f"\nAnswer: {answer}\n")
            except (ConnectionError, TimeoutError, ValueError, RuntimeError) as exc:
                print(f"Error during query: {exc}")


if __name__ == "__main__":
    main()
