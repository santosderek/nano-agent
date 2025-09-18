# MCP Progress Reporting Guidelines for Nano Agent

## When to Report Progress

### Always Report For:
- Operations taking longer than 1 second
- File operations over 1MB
- Directory operations with 100+ items
- Multi-step validation workflows
- Network or API calls
- Batch operations processing multiple items

### Progress Granularity:
- **Fine-grained** (every 1-5%): Critical operations, user-facing edits
- **Medium-grained** (every 10-20%): Standard file operations
- **Coarse-grained** (every 25-50%): Background tasks, validations

## Message Format Standards

### Status Messages Should Include:
1. **Action**: What is being done ("Reading", "Validating", "Writing")
2. **Target**: What is being operated on (file path, shortened if needed)
3. **Context**: Size, count, or other relevant metrics
4. **Stage**: Current phase in multi-step operations

### Example Messages:
- ✅ Good: "Reading configuration file (2.3KB)..."
- ✅ Good: "Processing file 15 of 42: data.json"
- ✅ Good: "Validation complete, starting execution phase"
- ❌ Bad: "Working..."
- ❌ Bad: "Processing stuff"

## Implementation Patterns

### Pattern 1: Simple Operation
```python
report_progress(0, 100, f"Reading {file_path}...")
# ... perform read ...
report_progress(100, 100, f"Successfully read {file_size} bytes")
```

### Pattern 2: Multi-Stage Operation
```python
stages = [
    (0, 20, "Initializing"),
    (20, 40, "Validating"),
    (40, 80, "Processing"),
    (80, 100, "Finalizing")
]

for start, end, stage in stages:
    report_progress(start, 100, f"{stage}...")
    # ... perform stage ...
```

### Pattern 3: Batch Processing
```python
total_items = len(items)
for i, item in enumerate(items):
    progress = int((i / total_items) * 100)
    report_progress(progress, 100, f"Processing {item.name} ({i+1}/{total_items})")
    # ... process item ...
```

### Pattern 4: Indeterminate Progress
```python
report_progress(None, None, "Waiting for external validation...")
# ... wait for response ...
report_progress(None, None, "Validation received, processing...")
```

## Error Reporting

When errors occur during operations:
1. Immediately report the error with context
2. Include the stage where error occurred
3. Provide actionable information if possible

```python
report_progress(current, 100, f"ERROR: Failed to write file - Permission denied at {path}")
```

## Performance Considerations

- **Throttle Updates**: Don't report more than once per 100ms
- **Batch Small Operations**: Group small file operations before reporting
- **Async When Possible**: Use async progress reporting for I/O operations
- **Memory Efficient**: Don't store entire progress history, just current state

## User Experience Guidelines

1. **Set Expectations**: Always indicate if operation will take time
2. **Be Specific**: Users prefer "Processing line 1523 of 5000" over "Processing..."
3. **Show Impact**: Include what will change ("Will modify 3 files")
4. **Confirm Completion**: Always send a final 100% complete message
5. **Handle Cancellation**: Report if operation was cancelled vs completed

## Integration with MCP

### Using progressToken:
```python
# Start operation with progress token
operation_id = start_operation_with_progress()

# Report progress with token
report_progress(
    progress=50,
    total=100,
    message="Processing halfway complete",
    token=operation_id
)

# Complete operation
complete_operation(operation_id)
```

### Multi-Operation Coordination:
When multiple operations run in parallel:
- Use unique progressTokens for each
- Prefix messages with operation context
- Report aggregate progress separately if needed

## Testing Progress Reporting

### Test Scenarios:
1. **Quick operations** (<1s): Should not spam progress
2. **Long operations** (>10s): Should show steady progress
3. **Failed operations**: Should report failure point clearly
4. **Cancelled operations**: Should indicate cancellation
5. **Parallel operations**: Should track independently

### Validation Checklist:
- [ ] Messages are human-readable and informative
- [ ] Progress percentage is accurate (never goes backwards)
- [ ] Final message confirms completion/failure
- [ ] Error messages include actionable context
- [ ] Updates are throttled appropriately
- [ ] Large operations show incremental progress
- [ ] Indeterminate operations show status updates

## Nano Agent Specific Implementation

### For NANO_AGENT_SYSTEM_PROMPT Operations:
```python
# File read operation
if file_size > 1024 * 1024:  # 1MB
    report_progress(0, 100, f"Reading large file {file_path} ({file_size/1024/1024:.1f}MB)...")
    # ... read in chunks with progress updates ...
    report_progress(100, 100, f"Successfully read {file_size} bytes")

# Directory listing
if item_count > 100:
    report_progress(0, 100, f"Listing directory with {item_count} items...")
    # ... process items with updates ...
    report_progress(100, 100, f"Listed {item_count} items")
```

### For REAL_AGENT_SYSTEM_PROMPT Operations:
```python
# Multi-stage file operation with external validation
stages = [
    (0, 10, "Preparing operation environment"),
    (10, 30, "Validating targets and permissions"),
    (30, 40, "Running safety checks"),
    (40, 80, "Executing operation"),
    (80, 95, "Verifying operation success"),
    (95, 100, "Finalizing and cleaning up")
]

for start, end, stage_name in stages:
    report_progress(start, 100, f"{stage_name}...")
    # ... perform stage operations ...
    if stage_error:
        report_progress(start, 100, f"ERROR at {stage_name}: {error_details}")
        break
```

## Context Integration

### With MCP Context (ctx):
```python
# In nano agent functions
async def prompt_nano_agent(agentic_prompt: str, model: str, provider: str, ctx: Any = None):
    if ctx:
        await ctx.report_progress(0.1, 1.0, "Initializing nano agent...")
        await ctx.report_progress(0.5, 1.0, "Executing agent operations...")
        await ctx.report_progress(1.0, 1.0, "Agent execution completed")
```

### Agent-Level Reporting:
```python
# In agent execution loops
class ProgressReportingAgent:
    def __init__(self, ctx=None):
        self.ctx = ctx

    async def report_progress(self, current, total, message):
        if self.ctx and hasattr(self.ctx, 'report_progress'):
            await self.ctx.report_progress(current, total, message)
        else:
            logger.info(f"Progress [{current}/{total}]: {message}")
```

## Real-World Examples

### Example 1: Large File Processing
```python
# Processing a 10MB file
await report_progress(0, 100, "Starting large file analysis (10.2MB)...")
await report_progress(25, 100, "Parsing file structure...")
await report_progress(50, 100, "Analyzing content patterns...")
await report_progress(75, 100, "Generating insights...")
await report_progress(100, 100, "Analysis complete - found 1,523 patterns")
```

### Example 2: Batch File Operations
```python
# Processing multiple files
files = get_files_to_process()
total_files = len(files)

await report_progress(0, 100, f"Starting batch processing of {total_files} files...")

for i, file in enumerate(files):
    progress = int((i / total_files) * 100)
    await report_progress(progress, 100, f"Processing {file.name} ({i+1}/{total_files})")

    # Process file...

await report_progress(100, 100, f"Batch processing complete - processed {total_files} files")
```

### Example 3: Multi-Stage Validation
```python
# Complex validation workflow
await report_progress(0, 100, "Starting comprehensive validation...")
await report_progress(10, 100, "Checking file permissions...")
await report_progress(30, 100, "Validating file integrity...")
await report_progress(50, 100, "Running security checks...")
await report_progress(70, 100, "Verifying external dependencies...")
await report_progress(90, 100, "Finalizing validation report...")
await report_progress(100, 100, "Validation complete - all checks passed")
```