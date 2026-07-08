import requests
from retrieval import retrieval

GROQ_API_KEY = ""
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"


def call_groq(messages, temperature=0):
    """Core function to handle the HTTP request."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {"model": MODEL, "messages": messages, "temperature": temperature}

    response = requests.post(GROQ_URL, headers=headers, json=payload)

    if response.status_code == 200:
        print(response.json())
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Request failed: {response.status_code} - {response.text}"


PROMPT = """

You are a Python Expert

Use only provided context.

Context :
    {context}

Question:
    {query}

Answer:
"""


def ask():
    user_query = input("Please enter the query")
    context = retrieval(user_query)
    prompt = PROMPT.format(context=context, query=user_query)


ask()
