# evaluate_ragbench.py

import json
from model import query_llm, compute_adherence, compute_relevance, compute_utilization, compute_completeness

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def evaluate_sample(sample, model, apikey):
    question = sample["question"]
    context = " ".join(sample["retrieved_contexts"])  # adjust key as needed
    ground_truth = sample.get("answer", "")

    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = query_llm(prompt, model=model, apikey=apikey)

    adherence = compute_adherence(response, context)
    relevance = compute_relevance(context, question)
    utilization = compute_utilization(context, response)
    completeness = compute_completeness(context, response)

    return {
        "question": question,
        "response": response,
        "adherence": adherence,
        "relevance": relevance,
        "utilization": utilization,
        "completeness": completeness
    }

def evaluate_file(filepath, model, apikey):
    data = load_data(filepath)
    results = []

    for i, sample in enumerate(data[:10]):  # use [:10] for testing
        print(f"Evaluating sample {i + 1}/{len(data)}")
        result = evaluate_sample(sample, model, apikey)
        results.append(result)

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Evaluation complete. Results saved to evaluation_results.json")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--model", type=str, default="llama3-8b-8192")
    parser.add_argument("--apikey", type=str, required=True)

    args = parser.parse_args()

    evaluate_file(args.file, args.model, args.apikey)
