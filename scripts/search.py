import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_intelligence.database.connection import create_db_engine
from global_intelligence.embeddings.encoder import EmbeddingEncoder
from global_intelligence.retrieval.pipeline import search


def main():
    encoder = EmbeddingEncoder()
    engine = create_db_engine()
    query_text = "what are the main financial risks?"
    final_results = search(query_text, encoder, engine)

    max_score = max(result["colbert_score"] for result in final_results)

    for result in final_results:
        print(f"Empresa: {result['company_name']} ({result['ticker']})")
        print(f"Relatório: {result['form_type']}")
        print(f"Fonte: {result['source']}")
        print(f"ColBERT Score: {result['colbert_score'] / max_score:.6f}")
        print(f"RRF Score: {result['score']:.6f}")
        print(f"Texto: {result['content'][:200]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
