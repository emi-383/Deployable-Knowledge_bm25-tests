from gliner import GLiNER

from core.rag.new_bm25 import getCorpus, testQuery, tokenizeCorpus, tokenizeQuery
# Reuse functions for now

'''
TO RUN
python -m core.rag.h_gliner-ent+bm25

gliner on query, bm25 rank
'''

#model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
model = GLiNER.from_pretrained("knowledgator/gliner-relex-base-v1.0")
#model = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v0.5")

def generate_labels(user_query):
    # Base labels used to break down any input sentence
    base_labels = ["Procedure", "Subject", "Equipment", "Technical Term", "Action", "Location", "Person"]

    # Extract entities from query
    query_entities = model.predict_entities(
        user_query, base_labels, threshold=0.2
    )
    gen_labels = [ent["text"] for ent in query_entities]

    # Fallback in case the query is too short or vaguely worded
    if not gen_labels:
        # Use basic words from query instead
        gen_labels = [word for word in user_query.split() if len(word) > 3]
    
    # Turn labels into string for bm25
    combined_keywords = " ".join(gen_labels)
    print(f"New labels: '{combined_keywords}'")
      
    return combined_keywords


def finalResults(results, scores, corpusText, corpusData):
    print("\n[hybrid GLiNER--v-ent -> bm25]")

    for i in range(results.shape[1]):
        doc_id = results[0, i]
        score = scores[0, i]

        print(f"\nRank {i+1} (BM25 Score: {score:.2f}): Document ID {doc_id}")
        print(f"Page number: {corpusData[doc_id]}")
        print(f"Snippet: {corpusText[doc_id][:300]}...")
        print("-" * 40)


def main():
    # Test
    query = "How do I perform a 9-line when reporting an injured soldier?"
    
    # Get corpus
    corpus, pageNum = getCorpus()
    corpusIndex = tokenizeCorpus(corpus)

    # Gliner reads the query to make new labels
    clean_keyword_string = generate_labels(query)
    # Make it readable for bm25
    query_tokens = tokenizeQuery(clean_keyword_string)

    # Bm25 rank based on gliner query and full corpus
    results, scores = corpusIndex.retrieve(query_tokens, k=10, sorted=True)
    finalResults(results, scores, corpus, pageNum)
    print("Done")


if __name__ == "__main__":
    main()
