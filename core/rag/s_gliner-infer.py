from gliner import GLiNER

from core.rag.new_bm25 import getCorpus
# Reuse for now

'''
TO RUN
python -m core.rag.s_gliner-infer

This file only for relex models
'''

model = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v0.5")
#model = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v1.0")
#model = GLiNER.from_pretrained("knowledgator/gliner-relex-base-v1.0")

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

    print(f"---New entity labels: {gen_entity_labels}")
    print(f"---New relation labels: {gen_relation_labels}")
    
    return gen_entity_labels, gen_relation_labels


def scan_corpus(corpusText, corpusData, target_entity_labels, target_relation_labels):
    found_matches = []

    # Iterate thru the document
    for idx, text_content in enumerate(corpusText):
        page_number = corpusData[idx]

        # Extract entities+relations from text
        entities_batch, relations_batch = model.inference(
            texts=[text_content], 
            labels=target_entity_labels, 
            relations=target_relation_labels, 
            threshold=0.4,
            adjacency_threshold=0.5,
            relation_threshold=0.7,
            return_relations=True
        )
        # Unpack the results for the current text snippet
        doc_entities = entities_batch[0]
        doc_relations = relations_batch[0]

        if doc_entities or doc_relations:
            # Group identical entities
            entity_mentions = [ent["text"] for ent in doc_entities]
            
            # Group relations into readable triplets: (Head -> Label -> Tail)
            relation_mentions = []
            relation_score = 0
            
            for rel in doc_relations:
                # Handle varying dictionary structures gracefully
                head_node = rel.get("head_text") or (rel.get("head", {}).get("text") if isinstance(rel.get("head"), dict) else str(rel.get("head", "Unknown")))
                tail_node = rel.get("tail_text") or (rel.get("tail", {}).get("text") if isinstance(rel.get("tail"), dict) else str(rel.get("tail", "Unknown")))
                label = rel.get("label", "related_to")
                
                relation_mentions.append(f"({head_node} -> {label} -> {tail_node})")
                relation_score += float(rel.get("score", 0.5))

            # Calculate a match score based on both extraction confidences
            total_ent_score = sum(float(ent["score"]) for ent in doc_entities)
            
            # Relations weighted heavier; can adjust
            total_score = total_ent_score + (relation_score * 2.0)

            found_matches.append(
                {
                    "doc_id": idx,
                    "page": page_number,
                    "score": total_score,
                    "entities_found": list(set(entity_mentions)),
                    "relations_found": list(set(relation_mentions)),
                    "snippet": text_content[:300],
                }
            )
    # Sort pages by combined extraction score
    found_matches.sort(key=lambda x: x["score"], reverse=True)
    return found_matches


def main():
    input_query = "How do I perform a 9-line when reporting an injured soldier?"

    # Get corpus
    corpus, pageNum = getCorpus()
   
    # Generate new target labels from user query; scan with new labels
    new_entity_labels, new_relation_labels = generate_labels(input_query)
    matches = scan_corpus(corpus, pageNum, new_entity_labels, new_relation_labels)

    print(f"\n[solo GLiNER-Relex--v-infer]")
    for rank, match in enumerate(matches[:10], 1):
        print(
            f"\nRank {rank} (Extraction Strength: {match['score']:.2f}) -> Doc ID {match['doc_id']}"
        )
        print(f"Page Number: {match['page']}")
        
        if match["relations_found"]:    ###
            print(f"Relations Detected: {match['relations_found']}")
        elif match["entities_found"]:
            print(f"Entities Detected: {match['entities_found']}")
            
        print(f"Snippet: {match['snippet']}...")
        print("-" * 50)


if __name__ == "__main__":
    main()