import numpy as np
from gliner import GLiNER

from core.rag.new_bm25 import getCorpus, tokenizeCorpus, tokenizeQuery, ranker
# Reuse stuff for now

'''
TO RUN
python -m core.rag.h2_bm25+gliner

This file only for relex models
'''

#model = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v0.5")
#model = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v1.0")
model = GLiNER.from_pretrained("knowledgator/gliner-relex-base-v1.0")

def generate_labels(user_query):    
    base_entity_labels = ["Procedure", "Subject", "Equipment", "Technical Term", "Location", "Person"]
    base_relation_labels = ["Action", "Interaction", "Process", "Dependency", "Requirement", "Verb", "Ownership"]
    all_base_labels = base_entity_labels + base_relation_labels

    # Extract entities+relations from query for use as new labels
    entities_batch, _ = model.inference(
        texts=[user_query], 
        labels=all_base_labels, 
        relations=[], 
        threshold=0.2
    )

    # Sort extracted labels into entity/relation containers
    query_spans = entities_batch[0]
    gen_entity_labels = []
    gen_relation_labels = []
    for span in query_spans:
        extracted_text = span["text"]
        matched_label = span["label"]
        
        if matched_label in base_entity_labels:
            gen_entity_labels.append(extracted_text)
        elif matched_label in base_relation_labels:
            gen_relation_labels.append(extracted_text)

    # Fallbacks in case query too short or vaguely worded
    if not gen_entity_labels:
        # Use basic words from query instead
        gen_entity_labels = [word for word in user_query.split() if len(word) > 3]
    if not gen_relation_labels:
        # Use base relation labels instead
        gen_relation_labels = ["related to", "involves", "requires", "verb"]

    # Remove duplicates
    gen_entity_labels = list(set(gen_entity_labels))
    gen_relation_labels = list(set(gen_relation_labels))

    print(f">>> New entity labels created: {gen_entity_labels}")
    print(f">>> New relation labels created: {gen_relation_labels}")
    
    return gen_entity_labels, gen_relation_labels


def main():
    input_query = "How do I perform a 9-line when reporting an injured soldier?"

    # Generate new target labels from user query
    new_entity_labels, new_relation_labels = generate_labels(input_query)
    
    # Get corpus & run BM25
    corpus, pageNum = getCorpus()
    corpusIndex = tokenizeCorpus(corpus)
    queryTokens = tokenizeQuery(input_query)
    
    # Store BM25 results in 2D array; can replace '_' with 'bm25_scores' if need
    bm25_results, _ = ranker(queryTokens, corpusIndex)

    # Flatten 2D into 1D list of integer IDs
    if isinstance(bm25_results, np.ndarray):
        top_doc_indices = bm25_results.flatten()
    elif isinstance(bm25_results, list) and isinstance(bm25_results[0], (list, np.ndarray)):
        top_doc_indices = np.array(bm25_results[0]).flatten()
    else:
        top_doc_indices = np.array(bm25_results).flatten()

    # Gliner reranks top results from bm25
    matches = []
    for doc_idx in top_doc_indices:
        # Pull text directly from original raw text
        doc_idx = int(doc_idx)
        page_text = str(corpus[doc_idx])
        page_no = pageNum[doc_idx]

        entities_batch, relations_batch = model.inference(
            texts=[page_text],              
            labels=new_entity_labels,       
            relations=new_relation_labels,  
            threshold=0.3,                  
            relation_threshold=0.4,         
            return_relations=True,
            flat_ner=True
        )
        page_entities = entities_batch[0] if entities_batch else []
        page_relations = relations_batch[0] if relations_batch else []

        # Compute extraction strength score
        relation_score = sum(float(r.get("score", 0.0)) for r in page_relations)
        entity_score = sum(float(e.get("score", 0.0)) for e in page_entities)
        combined_score = (relation_score * 2.0) + (entity_score * 0.5)

        matches.append({
            "doc_id": doc_idx,
            "page": page_no,
            "score": combined_score,
            "entities_found": page_entities,
            "relations_found": page_relations,
            "snippet": page_text[:300].replace('\n', ' ')
        })

    # Sort final matches by gliner score descending
    matches.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n[hybrid2 BM25 -> GLiNER-Relex--v-infer]")
    for rank, match in enumerate(matches[:10], 1):
        print(f"\nRank {rank} (Extraction Strength: {match['score']:.2f}) -> Doc ID {match['doc_id']}")
        print(f"Page Number: {match['page']}")
        
        if match["relations_found"]:    
            print(f"Relations Detected: {match['relations_found']}")
        if match["entities_found"]:
            print(f"Entities Detected: {match['entities_found']}")
            
        print(f"\nSnippet: {match['snippet']}...")
        print("-" * 50)


if __name__ == "__main__":
    main()