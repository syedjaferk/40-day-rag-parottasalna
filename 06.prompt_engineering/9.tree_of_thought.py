import re

import requests


def call_groq(prompt, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


def solve_with_tot(api_key, question):
    propose_prompt = f"""
    Question: {question}
    Provide 3 distinct possible first steps to solve this problem.
    Label them 'Step 1A:', 'Step 1B:', and 'Step 1C:'.
    """

    print(propose_prompt)
    input()

    branches = call_groq(propose_prompt, api_key)
    print("--- Generated Branches ---\n", branches)

    evaluate_prompt = f"""
    Here are 3 possible reasoning paths to solve: {question}

    {branches}

    Analyze each step for mathematical correctness. Which one is the most robust?
    Respond ONLY with the label (e.g., 'Step 1A') and a brief justification.
    """
    best_path_label = call_groq(evaluate_prompt, api_key)
    # print("\n--- Evaluator's Choice ---\n", best_path_label)

    print(evaluate_prompt)
    input()

    final_prompt = f"""
    Based on this winning strategy: {best_path_label}
    Complete the calculation for the question: {question}
    Provide the final answer.
    """

    print(final_prompt)
    input()

    return call_groq(final_prompt, api_key)


API_KEY = "YOUR_GROK_API_KEY"
result = solve_with_tot(
    API_KEY, "Find the sum of cube of all numbers between 1 and 50."
)
print("\n--- Final Output ---\n", result)
