import requests


def call_groq(messages):
    api_key = "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"

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": messages,
        "temperature": 0.0
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
