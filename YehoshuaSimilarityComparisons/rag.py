import os
# import openai
import random
# openai.api_key = 'REMOVED_LEAKED_KEY'
from embeddings_comparison import compare_sentences


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
if torch.cuda.is_available():
    device = "cuda"
    torch.device('cuda')
else:
    torch.device('cpu')
    device = "cpu"


def generate_queries(original_query):
    model = AutoModelForCausalLM.from_pretrained("dicta-il/dictalm2.0-instruct", torch_dtype=torch.bfloat16,
                                                 device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0-instruct")

    messages = [
        {"role": "user", "content": "איזה רוטב אהוב עליך?"},
        {"role": "assistant",
         "content": f"{original_query}"},
        {"role": "user", "content": "האם יש לך מתכונים למיונז?"}
    ]

    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    generated_ids = model.generate(encoded, max_new_tokens=50, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)

    generated_queries = decoded[0].strip().split("\n")
    return generated_queries


# Returns cosine similarity between query and documents
def vector_search(query, all_documents):
    available_docs = list(all_documents.keys())
    scores = {doc: compare_sentences([doc, query]) for doc in available_docs}
    return {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}


# Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    print("Initial individual search result ranks:")
    for query, doc_scores in search_results_dict.items():
        print(f"For query '{query}': {doc_scores}")

    for query, doc_scores in search_results_dict.items():
        for rank, (docc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            print(
                f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    print("Final reranked results:", reranked_results)
    return reranked_results


# Dummy function to simulate generative output
def generate_output(reranked_results, queries):
    return f"Final output based on {queries} and reranked documents: {list(reranked_results.keys())}"


# Predefined set of documents (usually these would be from your search database)
all_documents = {
    "doc1": "Climate change and economic impact.",
    "doc2": "Public health concerns due to climate change.",
    "doc3": "Climate change: A social perspective.",
    "doc4": "Technological solutions to climate change.",
    "doc5": "Policy changes needed to combat climate change.",
    "doc6": "Climate change and its impact on biodiversity.",
    "doc7": "Climate change: The science and models.",
    "doc8": "Global warming: A subset of climate change.",
    "doc9": "How climate change affects daily weather.",
    "doc10": "The history of climate change activism."
}

# Main function
if __name__ == "__main__":
    original_query = "impact of climate change"
    generated_queries = generate_queries(original_query)

    all_results = {}
    for query in generated_queries:
        search_results = vector_search(query, all_documents)
        all_results[query] = search_results

    reranked_results = reciprocal_rank_fusion(all_results)

    final_output = generate_output(reranked_results, generated_queries)

    print(final_output)