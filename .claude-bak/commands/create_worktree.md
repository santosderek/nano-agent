# Create Git Worktree

Create a new git worktree for an agent to work in isolation.

## Variables
worktree_name: $ARGUMENTS (IMPORTANT: first argument)
WORKTREE_PATH: trees

## Instructions

Execute ALL the following steps in sequence WITHOUT using a todo list:

1. **Parse the variables** - extract the worktree_name from the first argument
2. **List existing worktrees** to check for name conflicts
3. Create a new git worktree in the `trees/<worktree_name>` directory
4. Base the worktree on the main branch
5. Copy the `.env` file from the root directory to the worktree (if it exists)
6. Create an initial commit in the worktree to establish the branch
7. Report the successful creation of the worktree

## Git Worktree Setup

Execute these steps in order:

1. **List existing worktrees to avoid collisions**:
   ```bash
   git worktree list
   ```
   
   **WARNING**: DO NOT use any of the existing worktree names shown above! Each worktree must have a unique name.
   - Check if the proposed `<worktree_name>` appears in the list
   - If it does, STOP and report: "Worktree name '<worktree_name>' is already in use. Please choose a different name."

2. **Create the trees directory** if it doesn't exist:
   ```bash
   mkdir -p trees
   ```

3. **Check if worktree directory already exists**:
   - If `trees/<worktree_name>` already exists, report that it exists and stop
   - Otherwise, proceed with creation

4. **Create the git worktree**:
   ```bash
   git worktree add trees/<worktree_name> -b <worktree_name>
   ```

5. **Copy environment file** (if exists):
   ```bash
   if [ -f .env ]; then
     cp .env trees/<worktree_name>/.env
     echo "Copied .env file"
   else
     echo ".env file not found in root, skipping"
   fi
   ```

6. **Create initial commit with no changes**:
   ```bash
   git -C trees/<worktree_name> commit --allow-empty -m "Initial worktree setup for <worktree_name>"
   ```

## Error Handling

- **If the worktree name is already in use**, report this immediately and exit: "Worktree name '<worktree_name>' is already in use. Please choose a different name."
- If the worktree directory already exists, report this and exit gracefully
- If git worktree creation fails, report the error
- If .env doesn't exist in root, continue without error (it's optional)

## Verification

After setup, verify the worktree was created:
```bash
ls -la trees/<worktree_name>  # Should show full repository contents
git -C trees/<worktree_name> branch  # Should show the new branch
```

## Report

Report one of the following:
- Success: "Worktree '<worktree_name>' created successfully at trees/<worktree_name>"
- Already exists: "Worktree '<worktree_name>' already exists at trees/<worktree_name>"
- Error: "Failed to create worktree: <error message>"

## Notes

- Git worktrees provide branch isolation with separate working directories
- Each worktree has its own branch and can be worked on independently
- The worktree is created in the `trees/` directory to keep them organized
- Full repository contents are available in the worktree