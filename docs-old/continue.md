# Continue Configuration Management

The `cmc llm continue` command suite provides an interface for managing the configuration of the Continue AI extension. This allows you to inspect and eventually manage your LLM settings without manually navigating to the Continue configuration files.

## Overview

Continue stores its configuration in a YAML file located at:
`~/.continue/config.yaml`

This file defines the models used for different tasks (chat, autocomplete, edit, etc.), the providers (e.g., vLLM, OpenAI, Anthropic), and general extension settings.

## Usage

### 1. List Configured Models

You can view all currently configured models and their roles within Continue.

```bash
cmc llm continue list
```

This command parses the `config.yaml` and lists each model along with its associated roles. Roles typically include:
- `chat`: The primary model used for conversational interaction.
- `edit`: Used for inline code edits.
- `apply`: Used for applying changes to the codebase.
- `summarize`: Used for generating summaries of code blocks.
- `autocomplete`: The model used for ghost-text suggestions.

### Command Summary Table

| Command / Option | Description | Example |
| :--- | :--- | :--- |
| `list` | List all configured models and their roles | `cmc llm continue list` |
| `probe` | Show all values currently set via the GUI | `cmc llm continue probe` |
| `<profile>` | (Planned) Sync model from `llm.yaml` profile | `cmc llm continue uva.gemma` |

### 2. Planned Functionality: Profile-based Updates

Future versions of the `cmc llm continue` command will support synchronizing models directly from `llm.yaml` profiles, similar to the `cmc llm cline` command.

**Planned Workflow:**
1. Resolve a profile from `llm.yaml` (e.g., `uva.gemma`).
2. Extract the `model` and `apiBase`.
3. Update the `models` list in `config.yaml`, prioritizing the `chat` role.
4. Create an incremental backup of `config.yaml` before applying changes.

## Technical Details

### Configuration Format
Unlike Cline, which uses JSON, Continue uses YAML. The `ContinueManager` utilizes `PyYAML` to read and write these configurations to ensure the structure is preserved.

### Safety & Backups
Any operation that modifies the `config.yaml` file employs the `backup_name` utility from `cloudmesh-ai-common`. This creates incremental backups (e.g., `config.yaml.bak.1`) to prevent data loss during configuration updates.

### File Path
- **Config File**: `~/.continue/config.yaml`