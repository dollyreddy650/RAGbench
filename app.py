# app.py

import gradio as gr
from evaluate_ragbench import evaluate_file

def run_eval(file_obj, model_name, apikey):
    filepath = file_obj.name
    evaluate_file(filepath, model_name, apikey)
    with open("evaluation_results.json", "r") as f:
        results = f.read()
    return results

iface = gr.Interface(
    fn=run_eval,
    inputs=[
        gr.File(label="Upload RAGBench Dataset JSON"),
        gr.Textbox(label="Model Name (e.g., llama3-8b-8192)"),
        gr.Textbox(label="API Key", type="password")
    ],
    outputs="text",
    title="RAGBench Evaluation",
    description="Run RAGBench metrics (Adherence, Relevance, Utilization, Completeness)"
)

if __name__ == "__main__":
    iface.launch()
