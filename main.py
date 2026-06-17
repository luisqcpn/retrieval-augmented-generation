"""
Entry point for the RAG pipeline demo.

Run interactively:
    python main.py

Or pass a one-shot question via --query flag:
    python main.py --query "Quais são os direitos dos titulares de dados?"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline

load_dotenv()

DOCUMENTS_PATH = Path(__file__).parent / "documents" / "governance_docs.json"
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEMO_QUESTIONS = [
    "Quais são os direitos dos titulares de dados previstos na LGPD?",
    "Qual é o prazo para comunicar um incidente de segurança à ANPD?",
    "O que é Privacy by Design e quais são seus princípios?",
    "Quais são os prazos de retenção de dados de colaboradores?",
    "Quando é necessário elaborar um Relatório de Impacto (RIPD)?",
]


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "google", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def print_result(result: dict) -> None:
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"PERGUNTA: {result['question']}")
    print(separator)
    print(f"\nRESPOSTA:\n{result['answer']}")
    print(f"\nFONTES RECUPERADAS ({len(result['sources'])} documento(s)):")
    for src in result["sources"]:
        print(
            f"  • [{src['id']}] {src['title']} "
            f"(categoria: {src['category']}, similaridade: {src['similarity_score']:.4f})"
        )
    print(separator)


def run_interactive(pipeline: RAGPipeline) -> None:
    print("\n=== RAG Pipeline — Documentos de Governança e LGPD ===")
    print("Digite sua pergunta ou 'sair' para encerrar.\n")
    while True:
        try:
            question = input("Pergunta > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando pipeline.")
            break
        if not question:
            continue
        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando pipeline.")
            break
        result = pipeline.query(question)
        print_result(result)


def run_demo(pipeline: RAGPipeline) -> None:
    print("\n=== Executando perguntas de demonstração ===\n")
    all_results = []
    for question in DEMO_QUESTIONS:
        result = pipeline.query(question)
        print_result(result)
        all_results.append(result)

    output_path = Path("demo_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [{k: v for k, v in r.items()} for r in all_results],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResultados exportados para {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — Documentos de Governança e LGPD"
    )
    parser.add_argument("--query", "-q", help="Pergunta única (modo não-interativo)")
    parser.add_argument("--demo", action="store_true", help="Executar perguntas de demonstração")
    parser.add_argument("--top-k", type=int, default=3, help="Número de documentos a recuperar")
    parser.add_argument("--verbose", "-v", action="store_true", help="Logs detalhados (DEBUG)")
    parser.add_argument(
        "--output-json", help="Salvar resultado em JSON (apenas com --query)"
    )
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    try:
        pipeline = RAGPipeline(documents_path=str(DOCUMENTS_PATH), top_k=args.top_k)
    except EnvironmentError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if args.query:
        result = pipeline.query(args.query)
        print_result(result)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info("Result saved to %s", args.output_json)
    elif args.demo:
        run_demo(pipeline)
    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
