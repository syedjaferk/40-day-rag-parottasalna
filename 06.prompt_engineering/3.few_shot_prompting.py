import requests


def get_few_shot(api_key, query):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    messages = [
        {
            "role": "system",
            "content": "You are a mathematical assistant. Use the pairing method to find sums.",
        },
        # Example 1
        {"role": "user", "content": "Sum of even numbers 1-6?"},
        {"role": "assistant", "content": "Evens: 2, 4, 6. Calculation: 2+4+6 = 12."},
        # Example 2
        {"role": "user", "content": "Sum of even numbers 1-12?"},
        {
            "role": "assistant",
            "content": "Evens: 2, 4, 6, 8, 10, 12. Pairs: (2+12)+(4+10)+(6+8) = 14*3 = 42.",
        },
        # The Actual Task
        {"role": "user", "content": query},
    ]

    payload = {"model": "openai/gpt-oss-120b", "messages": messages, "temperature": 1}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


API_KEY = "YOUR_GROK_API_KEY"
print(get_few_shot(API_KEY, "Find the sum of all even numbers between 1 and 50."))
