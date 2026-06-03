import requests


def get_system_prompt(api_key, query):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": "You are a logic engine. You only respond in JSON format. Do not provide conversational filler. Key: 'result', Value: the final answer.",
            },
            {"role": "user", "content": query},
        ],
        "temperature": 0.0,
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


# Usage
API_KEY = "YOUR_GROK_API_KEY"
print(get_system_prompt(API_KEY, "Find the sum of all even numbers between 1 and 100."))
