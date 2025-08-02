import gradio as gr
from rgb_evaluation.evaluate import run_evaluation  # example

def evaluate_model(dataset, model_name):
    # dummy interface — adapt this to call your real logic
    result = run_evaluation(dataset, model_name)
    return result

iface = gr.Interface(
    fn=evaluate_model,
    inputs=[
        gr.Dropdown(choices=["en_refine.json", "en_int.json", "en_fact.json"], label="Dataset"),
        gr.Textbox(label="Model Name (e.g., llama3-8b-8192)")
    ],
    outputs="text",
    title="RGB Benchmark Evaluation",
    description="Evaluate LLMs using RGB metrics: Accuracy, Rejection, Factual Consistency"
)

if __name__ == "__main__":
    iface.launch()
