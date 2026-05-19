import os

from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

context = """
FastAPI is a modern Python web framework.
It supports async APIs.
"""

question = "What is FastAPI?"

prompt = f"""
Answer using the context below.

Context:
{context}

Question:
{question}
"""

response = client.chat.completions.create(
    model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
