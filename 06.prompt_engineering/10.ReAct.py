import json

import requests

# --- Config ---
API_KEY = "YOUR_GROK_API_KEY"
URL = "https://api.groq.com/openai/v1/chat/completions"


# --- Simulated Tools ---
def get_weather(city):
    # In a real app, this would hit a Weather API
    data = {"London": "15°C and Rainy", "Chennai": "32°C and Sunny"}
    return data.get(city, "Weather data not found.")


def run_react_agent(user_query):
    # 1. THE SYSTEM PROMPT (The "ReAct Instruction")
    system_prompt = """
    You are a ReAct Agent. You solve problems by alternating between Thought, Action, and Observation.
    Available tools:
    - get_weather[city]: Returns the current weather.

    Format:
    Thought: [Your reasoning]
    Action: [tool_name][argument]
    Observation: [Result from tool]
    ... (repeat until done)
    Final Answer: [The result]
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    print(f"User: {user_query}\n")

    # Limit loops to prevent infinite logic (Master practice)
    for _ in range(3):
        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": messages,
            "temperature": 0,
        }
        response = requests.post(
            URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload
        )
        ai_text = response.json()["choices"][0]["message"]["content"]

        print(ai_text)

        # Check if model wants to take an action
        if "Action:" in ai_text:
            # Parse the tool name and argument (Simple regex-like logic)
            # Example: Action: get_weather[Chennai]
            tool_call = ai_text.split("Action:")[1].strip()
            tool_name = tool_call.split("[")[0]
            print("Tool Name", tool_name)
            arg = tool_call.split("[")[1].split("]")[0]
            # print("Args ", arg) # Arguments
            input()  # for stopping
            # Execute the tool
            if tool_name == "get_weather":
                obs = get_weather(arg)
            else:
                obs = "Unknown tool."

            # Feed the Observation back into the conversation
            print(f"Observation: {obs}\n")
            messages.append({"role": "assistant", "content": ai_text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        else:
            # If no Action, the model has reached the Final Answer
            break


# --- Execution ---
run_react_agent("What is the weather in chennai ?")
