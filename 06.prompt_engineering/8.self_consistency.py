import re
from collections import Counter

import requests


def get_consistency_sample(api_key, query):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # We use a higher temperature to allow for diverse reasoning paths
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": "Solve the problem step-by-step. End your response with 'Final Answer: [number]'",
            },
            {"role": "user", "content": query},
        ],
        "temperature": 0.7,
    }

    response = requests.post(url, headers=headers, json=payload)
    content = response.json()["choices"][0]["message"]["content"]

    match = re.search(r"Final Answer:\s*(\d+)", content)
    return match.group(1) if match else None


def solve_with_self_consistency(api_key, query, num_samples=5):
    print(f"Generating {num_samples} reasoning paths...")
    results = []

    for i in range(num_samples):
        answer = get_consistency_sample(api_key, query)
        if answer:
            results.append(answer)
            print(f" Sample {i + 1}: {answer}")

    # Determine the majority vote
    if not results:
        return "No valid answers found."

    vote_count = Counter(results)
    final_answer, count = vote_count.most_common(1)[0]

    return f"\nConsistent Answer: {final_answer} (Found in {count}/{num_samples} paths)"


API_KEY = "YOUR_GROK_API_KEY"
question = "Find the sum of all even numbers between 1 and 50."
print(solve_with_self_consistency(API_KEY, question))
