# model.py

import requests
import difflib

# LLM Query Function using Groq API

def query_llm(prompt: str, model: str, apikey: str, temperature: float = 0.7) -> str:
    """Query the LLM using Groq's OpenAI-compatible API"""
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {apikey}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]


# TRACe Metric Approximations (simplified)


def compute_adherence(response: str, context: str) -> bool:
    """Boolean: Is the entire response grounded in the context?"""
    return all(sentence.strip() in context for sentence in response.split('.') if sentence.strip())

def compute_relevance(context: str, question: str) -> float:
    """Approximate relevance score (based on string similarity)"""
    return difflib.SequenceMatcher(None, question.lower(), context.lower()).ratio()

def compute_utilization(context: str, response: str) -> float:
    """Approximate utilization score based on how much of the context is used in the response"""
    context_tokens = set(context.lower().split())
    response_tokens = set(response.lower().split())
    overlap = context_tokens.intersection(response_tokens)
    return len(overlap) / max(len(context_tokens), 1)

def compute_completeness(context: str, response: str) -> float:
    """Approximate completeness: how much of relevant content is used in response"""
    relevant_points = [s for s in context.split('.') if len(s.split()) > 3]
    used = sum(1 for s in relevant_points if s.strip() in response)
    return used / max(len(relevant_points), 1)

