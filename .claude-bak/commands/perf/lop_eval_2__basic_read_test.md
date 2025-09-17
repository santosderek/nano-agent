# Problem #2: Basic File Reading Test

## Instructions

- Pass the prompt into each nano-agent AS IS. Do not change the prompt in any way.
- Each agent should execute the task independently.

## Variables

PROMPT: "Read the README.md file. Provide exactly the first 10 lines and the last 10 lines of the file. Format your response as follows:
FIRST 10 LINES:
[lines here]

LAST 10 LINES:
[lines here]

Respond with your entire JSON response structure as is."

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

Read the first and last 10 lines of the README.md file yourself and grade the agents based on the `Grading rubric` below.

IMPORTANT: All agents must will respond with this JSON structure. Don't change the structure or add any additional fields. Output it as the given structure as raw JSON for each agent with no preamble.

```json
{
    "success": true,
    "result": "<the formatted first and last 10 lines>",
    "error": null,
    "metadata": {
        ...keep all fields given
    },
    "execution_time_seconds": X.XX
}
```

## Grading rubric

- Did the agent correctly identify and return the first and last 10 lines? How close?
- Did the agent follow the specified format exactly? How well?