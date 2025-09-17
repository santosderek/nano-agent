---
allowed-tools: Read,Bash,Edit,Write
description: Converts absolute paths in .claude/settings.json command scripts to relative paths
---

# convert_paths_relative

This command converts all absolute paths in .claude/settings.json command scripts to relative paths. It helps make the configuration more portable and easier for others to use as a starter template.

## Instructions
- Get the current working directory using pwd
- Read the .claude/settings.json file
- Parse the JSON content to find all command scripts
- Identify absolute paths in the command scripts (paths that start with /)
- Convert each absolute path that points to files within the current project to a relative path
- Update the settings.json file with the converted relative paths
- Show the user what changes were made
- Handle cases where paths might be arguments to commands (e.g., python /absolute/path/script.py)
- Preserve the existing JSON formatting and structure
- Create a backup of the original settings.json before making changes