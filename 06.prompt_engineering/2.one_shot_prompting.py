import requests


def get_one_shot(api_key, query):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    messages = [
        {
            "role": "system",
            "content": "You solve math problems by listing the numbers first, then providing the sum.",
        },
        # The One Shot Example
        {
            "role": "user",
            "content": "Find the sum of all even numbers between 1 and 10.",
        },
        {"role": "assistant", "content": "Numbers: 2, 4, 6, 8, 10. Sum: 30."},
        # The Actual Task
        {"role": "user", "content": query},
    ]

    payload = {"model": "openai/gpt-oss-120b", "messages": messages, "temperature": 0.0}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


API_KEY = "YOUR_GROK_API_KEY"

print(get_one_shot(API_KEY, "Give me sum of all even numbers between 1 and 50."))
