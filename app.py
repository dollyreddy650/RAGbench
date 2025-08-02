# app.py

import gradio as gr
from ragbench_eval import evaluate_model

def run_eval(dataset_path, model_name, api_key):
    results = evaluate_model(dataset_path, model_name, api_key)
    return "\n\n".join([f"Q: {r['question']}\nA: {r['generated']}" for r in results[:3]])

iface = gr.Interface(
    fn=run_eval,
    inputs=[
        gr.Textbox(label="Dataset path (e.g., data/en_refine.json)"),
        gr.Textbox(label="Model name (e.g., llama3-8b-8192)"),
        gr.Textbox(label="Groq API Key", type="password")
    ],
    outputs="text",
    title="RAGBench Evaluation",
    description="Evaluate models on RAGBench tasks using Groq API"
)

if __name__ == "__main__":
    iface.launch()
