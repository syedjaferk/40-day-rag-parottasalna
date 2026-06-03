import requests

GROQ_API_KEY = "YOUR_GROK_API_KEY"
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


def get_zero_shot(query):
    """General prompting with no examples."""

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": query},
    ]
    return call_groq(messages)


if __name__ == "__main__":
    response = get_zero_shot("give me some details on python ")
    print(response)
