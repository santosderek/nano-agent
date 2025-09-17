# Problem #3: File Operations Test

## Instructions

- Pass the prompt into each nano-agent AS IS, replacing MODEL_NAME with the actual model name.
- Each agent works in isolation by using their unique model name in file paths.

## Variables

PROMPT: "Complete the following tasks:
1. Read the .claude/settings.json file and extract all unique hook names (the keys of the 'hooks' object).
2. Create a file called 'summary_MODEL_NAME.json' (replace MODEL_NAME with your actual model name) with the following content:
   {
     'model': 'MODEL_NAME',
     'hook_names': [<array of all unique hook names you found>],
     'test_status': 'completed'
   }
3. Create another file called 'signature_MODEL_NAME.txt' with the content: 'MODEL_NAME was here - Successfully completed file operations test'
4. List the current directory to show your created files exist.

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

Verify each agent created their uniquely named files and that the contents match the specifications.

IMPORTANT: All agents must will respond with this JSON structure. Don't change the structure or add any additional fields. Output it as the given structure as raw JSON for each agent with no preamble.

```json
{
    "success": true,
    "result": "<summary of files created and directory listing>",
    "error": null,
    "metadata": {
        ...keep all fields given
    },
    "execution_time_seconds": X.XX
}
```

## Grading rubric

- Did the agent correctly extract all unique hook names from .claude/settings.json? ["PreToolUse", "PostToolUse", "Notification", "Stop", "SubagentStop", "PreCompact", "UserPromptSubmit", "SessionStart"]
- Did the agent create both files with correct JSON structure and content?
- Did the agent use unique file names to avoid conflicts with other agents?
- Did the agent complete all 4 tasks successfully?