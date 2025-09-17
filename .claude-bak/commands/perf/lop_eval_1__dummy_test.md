# Problem #1: The Dummy Test

## Instructions

- Pass the prompt into each nano-agent AS IS. Do not change the prompt in any way.

## Variables

PROMPT: "What is the capital of the United States? Respond with your entire JSON response structure as is."

## Agents

IMPORTANT: You're calling the respective claude code sub agents - do not call the `mcp__nano-agent__prompt_nano_agent` tool directly, let the sub agent's handle that.

@agent-nano-agent-gpt-5-nano PROMPT
@agent-nano-agent-gpt-5-mini PROMPT
@agent-nano-agent-gpt-5 PROMPT
@agent-nano-agent-claude-opus-4-1 PROMPT
@agent-nano-agent-claude-opus-4 PROMPT
@agent-nano-agent-claude-sonnet-4 PROMPT
@agent-nano-agent-claude-3-haiku PROMPT
@agent-nano-agent-gpt-oss-20b PROMPT
@agent-nano-agent-gpt-oss-120b PROMPT

## Expected Output

The expected output for each agent is "Washington, D.C."

IMPORTANT: All agents must will respond with this JSON structure. Don't change the structure or add any additional fields. Output it as the given structure as raw JSON for each agent with no preamble.

```json
{
    "success": true,
    "result": "Washington, D.C.",
    "error": null,
    "metadata": {
        ...keep all fields given
    },
    "execution_time_seconds": X.XX
}
```

## Grading rubric

Here we're looking for the shortest most concise response. Grade accordingly.
