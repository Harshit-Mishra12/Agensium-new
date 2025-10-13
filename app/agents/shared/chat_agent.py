import openai
import json
import os
import time
from datetime import datetime, timezone
from fastapi import HTTPException

AGENT_VERSION = "1.1.0"  # Version for the chat agent

# Securely get the API key from environment variables (loaded from .env by main.py)
api_key = os.getenv("OPENAI_API_KEY")
# api_key="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"
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")

client = openai.OpenAI(api_key=api_key)

MODEL_NAME = 'gpt-4o-mini'  # Using OpenAI's powerful and cost-effective model

def answer_question_on_report(agent_report: dict, user_question: str, history: list = None):
    """
    Uses an OpenAI LLM to answer a user's question, considering the chat history for context.
    """
    start_time = time.time()
    
    try:
        # --- Prompt Engineering with History ---
        system_prompt = """
        You are 'Agensium Co-Pilot', a world-class AI data analyst. 
        Your sole purpose is to answer questions about a data analysis report provided in JSON format.
        You must adhere to the following rules:
        1. Base your answers *exclusively* on the information within the provided JSON report.
        2. Do not make up information, guess, or infer data that isn't present.
        3. If the answer cannot be found in the report, you must state that clearly.
        4. Answer concisely and directly in a helpful, professional tone.
        5. Use the provided chat history to understand the context of follow-up questions.
        """
        
        report_json_string = json.dumps(agent_report, indent=2)

        # The user prompt always includes the full report for context, even in follow-up questions.
        user_prompt_content = f"""
        Here is the data analysis report:
        ```json
        {report_json_string}
        ```

        Based ONLY on the report above, please answer the following question: "{user_question}"
        """

        # --- Construct the messages list for the API call ---
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add the previous conversation history if it exists
        if history:
            messages.extend(history)
            
        # Add the current user question
        messages.append({"role": "user", "content": user_prompt_content})
        
        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )

        answer = response.choices[0].message.content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get response from OpenAI. Error: {str(e)}")

    return {
        "agent": "ChatAgent",
        "results": {
            "status": "success",
            "user_question": user_question,
            "answer": answer
        }
    }

