import requests


def get_contextual_prompt(api_key, query):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Situational context provided to the model
    context = """
    Context: The user is learning about Gaussian Summation.
    The formula for the sum of the first 'n' even numbers is n(n + 1).
    In the range 1-50, there are exactly 25 even numbers.
    """

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": "Use the provided context to answer the user's question accurately.",
            },
            {"role": "user", "content": f"{context}\n\nQuestion: {query}"},
        ],
        "temperature": 0.0,
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


API_KEY = "YOUR_GROK_API_KEY"
print(
    get_contextual_prompt(API_KEY, "Find the sum of all even numbers between 1 and 50.")
)
