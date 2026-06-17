import numpy as np
from sentence_transformers import CrossEncoder

from core.rag.new_bm25 import getCorpus, tokenizeCorpus, tokenizeQuery, ranker

'''
TO RUN
python -m core.rag.h_bm25+RR

bert-based rerankers
'''
#model = "mixedbread-ai/mxbai-rerank-large-v1"   #heavy
#model  = "BAAI/bge-reranker-large"              #heavy, standard
model = "cross-encoder/ms-marco-MiniLM-L6-v2"   #light
model = CrossEncoder(model)

def main():
    input_query = "How do I perform a 9-line when reporting an injured soldier?"

    # Get corpus & run bm25
    corpus, pageNum = getCorpus()
    corpusIndex = tokenizeCorpus(corpus)
    queryTokens = tokenizeQuery(input_query)
    bm25_results, _ = ranker(queryTokens, corpusIndex)

    # Flatten 2D into 1D list of integer IDs
    if isinstance(bm25_results, np.ndarray):
        top_doc_indices = bm25_results.flatten()
    elif isinstance(bm25_results, list) and isinstance(bm25_results[0], (list, np.ndarray)):
        top_doc_indices = np.array(bm25_results[0]).flatten()
    else:
        top_doc_indices = np.array(bm25_results).flatten()

    # Rerank
    matches = []
    for doc_idx in top_doc_indices:
        doc_idx = int(doc_idx)
        page_text = str(corpus[doc_idx])
        page_no = pageNum[doc_idx]

        # Cross encode score
        semantic_score = model.predict([input_query, page_text])

        matches.append({
            "doc_id": doc_idx,
            "page": page_no,
            "score": float(semantic_score),
            "snippet": page_text[:300].replace('\n', ' ')
        })

    # Sort final matches by the reranked score descending
    matches.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n[hybrid BM25 -> ]")
    for rank, match in enumerate(matches[:10], 1):
        print(f"\nRank {rank} (Score: {match['score']:.4f}) - Doc ID {match['doc_id']}")
        print(f"Page Number: {match['page']}")
        print(f"Snippet: {match['snippet']}...")
        print("-" * 50)


if __name__ == "__main__":
    main()