import json

import requests


def get_chain_of_thought(api_key, query):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    messages = [
        {
            "role": "system",
            "content": "You are a logical reasoning engine. For every problem, show your work step-by-step before providing the final answer.",
        },
        # Few-Shot CoT Example
        {
            "role": "user",
            "content": "Find the sum of all even numbers between 1 and 10.",
        },
        {
            "role": "assistant",
            "content": """Step 1: Identify the even numbers in the range [1, 10]. They are 2, 4, 6, 8, 10.
Step 2: Group them into pairs for easier addition: (2 + 10) = 12, (4 + 8) = 12, and the middle number 6.
Step 3: Add the pairs and the remaining number: 12 + 12 + 6 = 30.
Final Answer: 30.""",
        },
        # The Actual Task
        {"role": "user", "content": query},
    ]

    payload = {"model": "openai/gpt-oss-120b", "messages": messages, "temperature": 0.0}

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


# Usage
API_KEY = "YOUR_GROK_API_KEY"
user_query = "Find the sum of all even numbers between 1 and 50."
print(get_chain_of_thought(API_KEY, user_query))
