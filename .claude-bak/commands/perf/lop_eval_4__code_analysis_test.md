# Problem #4: Code Analysis and Enhancement Test

## Instructions

- Pass the prompt into each nano-agent AS IS, replacing MODEL_NAME with the actual model name.
- Each agent creates their own enhanced version of the code.

## Variables

PROMPT: "Perform the following code engineering tasks:
1. Read the file at 'apps/nano_agent_mcp_server/src/nano_agent/modules/constants.py'
2. Analyze the code structure and identify:
   - Total number of constants defined
   - Number of different constant categories (models, errors, success messages, etc.)
   - The default model and provider values
3. Create a new Python file called 'analysis_MODEL_NAME.py' (replace MODEL_NAME with your actual model name) that contains:
   - A docstring with your analysis summary
   - A function called 'get_constants_report()' that returns a dictionary with your findings
   - A comment at the bottom: '# Analysis completed by MODEL_NAME'
4. Create another file 'enhanced_constants_MODEL_NAME.py' that adds one new useful constant:
   - Add: MODEL_SIGNATURE = 'Enhanced by MODEL_NAME'
   - Include all original constants plus your addition
5. Return a summary of your analysis and the paths to both files you created.

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

Verify each agent correctly analyzed the constants file, created both Python files, and provided an accurate summary.

IMPORTANT: All agents must will respond with this JSON structure. Don't change the structure or add any additional fields. Output it as the given structure as raw JSON for each agent with no preamble.

```json
{
    "success": true,
    "result": "<analysis summary and file paths created>",
    "error": null,
    "metadata": {
        ...keep all fields given
    },
    "execution_time_seconds": X.XX
}
```

## Grading rubric

- Did the agent correctly identify all constants and categories in their analysis?
- Did the agent create valid, well-structured Python files?
- Did the agent complete all 5 subtasks successfully?