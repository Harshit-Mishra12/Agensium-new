from email import message
import openai
import json
import os
import time
from datetime import datetime, timezone
from fastapi import HTTPException

AGENT_VERSION = "1.1.0"  # Version for the chat agent

# Securely get the API key from environment variables (loaded from .env by main.py)
# api_key = os.getenv("OPENAI_API_KEY")
api_key = "sk-"
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")

client = openai.OpenAI(api_key=api_key)

MODEL_NAME = 'gpt-4o-mini'  # Using OpenAI's powerful and cost-effective model

def answer_question_on_report(agent_report: dict, user_question: str, history: list = None):
    """
    Uses an OpenAI LLM to answer a user's question, considering the chat history for context.
    """
    start_time = time.time()
    history = None
    
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


def generate_llm_summary(agent_name: str, results: dict, audit_trail: dict) -> str:
    """
    Generate an LLM-powered summary of agent results.
    Shared helper function used by all data profiling agents.
    
    Args:
        agent_name: Name of the agent (e.g., "UnifiedProfiler", "RiskScorer")
        results: The results dictionary from the agent
        audit_trail: The audit trail dictionary from the agent
    
    Returns:
        str: A concise summary of findings and recommendations
    """  
    try:
        # Prepare focused data for the LLM
        summary_data = {
            "agent": agent_name,
            "scores": audit_trail.get("scores", {}),
            "findings_count": len(audit_trail.get("findings", [])),
            "critical_findings": [f for f in audit_trail.get("findings", []) if f.get("severity") == "critical"],
            "warning_findings": [f for f in audit_trail.get("findings", []) if f.get("severity") == "warning"],
            "high_findings": [f for f in audit_trail.get("findings", []) if f.get("severity") == "high"],
            "actions": audit_trail.get("actions", [])
        }
        
        system_prompt = f"""
        You are an expert data analyst assistant. Analyze the {agent_name} results and provide a concise summary.
        
        Your summary should:
        1. Highlight 2-3 most important findings
        2. Provide clear, actionable recommendations
        3. Be concise (3-5 sentences maximum)
        4. Use professional language
        5. Focus on what matters most
        """
        
        user_prompt = f"""
        Analysis data:
        {json.dumps(summary_data, indent=2)}
        
        Provide a concise summary of what was found and what is recommended.
        """
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Summary generation failed: {str(e)}. Please review the detailed findings in the audit trail."
