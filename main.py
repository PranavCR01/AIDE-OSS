import argparse
import os
import sys

# Ensure relative imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser(description="AIDE-OSS: AI-Driven Log Analyzer")

    parser.add_argument("--parse", action="store_true", help="Parse raw logs, detect anomalies, and generate summaries")
    parser.add_argument("--build-index", action="store_true", help="Rebuild FAISS index from summaries")
    parser.add_argument("--chat", action="store_true", help="Chat with logs using RAG pipeline")
    parser.add_argument("--evaluate-embeddings", action="store_true", help="Compare multiple embedding models")
    parser.add_argument("--benchmark-models", action="store_true", help="Run summarization model benchmark")
    parser.add_argument("--insight-summary", action="store_true", help="Generate anomaly insight summaries via Mistral")

    args = parser.parse_args()

    if args.parse:
        from pipeline.log_pipeline import run_log_pipeline
        run_log_pipeline()

    elif args.build_index:
        from pipeline.embedding_pipeline import build_faiss_index
        build_faiss_index()

    elif args.chat:
        from chat.chatbot import RAGChatbot
        chatbot = RAGChatbot()
        chatbot.chat_loop()

    elif args.evaluate_embeddings:
        from evaluation.embedding_evaluator import evaluate_all
        evaluate_all()

    elif args.benchmark_models:
        from evaluation.model_benchmark import run_benchmark
        run_benchmark()

    elif args.insight_summary:
        from agents.insight_agent import run_summary
        run_summary()

    else:
        print("\n Please specify one of the following:")
        print("   --parse | --build-index | --chat | --evaluate-embeddings | --benchmark-models | --insight-summary")

if __name__ == "__main__":
    main()
