#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "openai",
#     "python-dotenv",
# ]
# ///

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import constants
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import OPENAI_MODEL


def prompt_llm(prompt_text):
    """
    Base OpenAI LLM prompting method using fastest model.

    Args:
        prompt_text (str): The prompt to send to the model

    Returns:
        str: The model's response text, or None if error
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # GPT-5 models require specific parameters
        params = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_completion_tokens": 100,
        }
        
        # Add GPT-5 specific parameters
        if "gpt-5" in OPENAI_MODEL:
            params["reasoning_effort"] = "minimal"  # Fastest mode, no CoT
            params["verbosity"] = "low"
            # GPT-5 models only support default temperature (1)
        else:
            # Non-GPT-5 models use max_tokens and can use custom temperature
            params["max_tokens"] = params.pop("max_completion_tokens")
            params["temperature"] = 0.7
            
        response = client.chat.completions.create(**params)

        content = response.choices[0].message.content
        if content:
            return content.strip()
        else:
            if __name__ == "__main__":
                print(f"Debug: Empty response from model", file=sys.stderr)
            return None

    except Exception as e:
        if __name__ == "__main__":
            print(f"Debug: {str(e)}", file=sys.stderr)
        return None


def generate_completion_message():
    """
    Generate a completion message using OpenAI LLM.

    Returns:
        str: A natural language completion message, or None if error
    """
    engineer_name = os.getenv("ENGINEER_NAME", "").strip()

    if engineer_name:
        name_instruction = f"Sometimes (about 30% of the time) include the engineer's name '{engineer_name}' in a natural way."
        examples = f"""Examples of the style: 
- Standard: "Work complete!", "All done!", "Task finished!", "Ready for your next move!"
- Personalized: "{engineer_name}, all set!", "Ready for you, {engineer_name}!", "Complete, {engineer_name}!", "{engineer_name}, we're done!" """
    else:
        name_instruction = ""
        examples = """Examples of the style: "Work complete!", "All done!", "Task finished!", "Ready for your next move!" """

    prompt = f"""Generate a short, friendly completion message for when an AI coding assistant finishes a task. 

Requirements:
- Keep it under 10 words
- Make it positive and future focused
- Use natural, conversational language
- Focus on completion/readiness
- Do NOT include quotes, formatting, or explanations
- Return ONLY the completion message text
{name_instruction}

{examples}

Generate ONE completion message:"""

    response = prompt_llm(prompt)

    # Clean up response - remove quotes and extra formatting
    if response:
        response = response.strip().strip('"').strip("'").strip()
        # Take first line if multiple lines
        response = response.split("\n")[0].strip()

    return response


def main():
    """Command line interface for testing."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--completion":
            message = generate_completion_message()
            if message:
                print(message)
            else:
                print("Error generating completion message")
        else:
            prompt_text = " ".join(sys.argv[1:])
            response = prompt_llm(prompt_text)
            if response:
                print(response)
            else:
                print("Error calling OpenAI API")
    else:
        print("Usage: ./oai.py 'your prompt here' or ./oai.py --completion")


if __name__ == "__main__":
    main()