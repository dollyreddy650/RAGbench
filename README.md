# RAGBench Evaluation Framework

This repository contains an implementation of a Retrieval-Augmented Generation (RAG) pipeline and its evaluation using the [RAGBench dataset](https://huggingface.co/datasets/rungalileo/ragbench).

## 📌 Objective

Develop a real-world RAG system that:
- Accurately answers queries using retrieved documents
- Supports domain-specific adaptation without model finetuning
- Evaluates context relevance, utilization, completeness, and answer faithfulness

## 📚 Dataset

**RAGBench** is a large-scale benchmark dataset for RAG systems, spanning 100K+ examples from five real-world domains:
- Biomedical Research
- Legal
- General Knowledge
- Finance
- Customer Support

📎 [RAGBench on Hugging Face](https://huggingface.co/datasets/rungalileo/ragbench)

## 🛠️ Components

- Query classification
- Document retrieval (via vector store)
- Reranking and filtering
- Answer generation (LLMs like LLaMA, Kimi, Qwen)
- Summarization

## 📈 Evaluation Metrics

- **Context Relevance**
- **Context Utilization**
- **Answer Faithfulness**
- **Answer Completeness**

## ⚙️ Tech Stack

- `Langchain`, `LlamaIndex`, `Transformers`, `PyTorch`
- `FAISS` / `ChromaDB`
- `OpenAI`, `Grok`, `Llama`, `Moonshot`, `Qwen` APIs
- `FastAPI`, `Streamlit`, `Google Cloud` for deployment

## 🚀 Run Instructions

```bash
# Clone and install
git clone https://github.com/your-username/ragbench-eval.git
cd ragbench-eval
pip install -r requirements.txt

# Launch evaluation script
python evaluate_ragbench.py --model llama3-8b-8192 --apikey $GROK_API_KEY

