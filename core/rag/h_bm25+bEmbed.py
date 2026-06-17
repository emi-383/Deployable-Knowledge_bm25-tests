import numpy as np
from sentence_transformers import SentenceTransformer, util

from core.rag.new_bm25 import getCorpus, tokenizeCorpus, tokenizeQuery, ranker
# Reuse stuff for now

'''
TO RUN
python -m core.rag.h_bm25+ST
'''
model = SentenceTransformer("all-MiniLM-L6-v2")  #embed

def main():
    print("hi :D")
    input_query = "How do I perform a 9-line when reporting an injured soldier?"

    # Get corpus
    corpus, pageNum = getCorpus()
    corpusIndex = tokenizeCorpus(corpus)
    queryTokens = tokenizeQuery(input_query)
    
    # Run bm25 & store results
    bm25_results, _ = ranker(queryTokens, corpusIndex)
    # Flatten 2D results into 1D list of integer IDs
    if isinstance(bm25_results, np.ndarray):
        top_doc_indices = bm25_results.flatten()
    elif isinstance(bm25_results, list) and isinstance(bm25_results[0], (list, np.ndarray)):
        top_doc_indices = np.array(bm25_results[0]).flatten()
    else:
        top_doc_indices = np.array(bm25_results).flatten()

    # Sentence transformers encode query
    query_embedding = model.encode(input_query, convert_to_tensor=True)

    matches = []
    for doc_idx in top_doc_indices:
        # Pull text directly from corpus
        doc_idx = int(doc_idx)
        page_text = str(corpus[doc_idx])
        page_no = pageNum[doc_idx]

        # Encode top results from bm25
        doc_embedding = model.encode(page_text, convert_to_tensor=True)

        # Calc score
        semantic_score = util.cos_sim(query_embedding, doc_embedding).item()

        matches.append({
            "doc_id": doc_idx,
            "page": page_no,
            "score": semantic_score,
            "snippet": page_text[:300].replace('\n', ' ')
        })

    # Sort final matches by score descending
    matches.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n[hybrid BM25 -> sentence transformers ()]")
    for rank, match in enumerate(matches[:10], 1):
        print(f"\nRank {rank} (Score: {match['score']:.4f}) -> Doc ID {match['doc_id']}")
        print(f"Page Number: {match['page']}")
        print(f"Snippet: {match['snippet']}...")
        print("-" * 50)


if __name__ == "__main__":
    main()