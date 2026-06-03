import requests


def get_role_prompt(api_key, query):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": "You are a Senior Mathematics Professor who specializes in Arithmetic Progressions. Explain the formula used before giving the answer.",
            },
            {"role": "user", "content": query},
        ],
        "temperature": 0.3,
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


# Usage
API_KEY = "YOUR_GROK_API_KEY"
print(get_role_prompt(API_KEY, "Find the sum of all even numbers between 1 and 50."))
