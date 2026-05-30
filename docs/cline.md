# Deep Dive: Cline Setup & Configuration Management

The `cmc llm cline` command suite provides a powerful interface for managing the configuration of the Cline AI agent. Instead of manually editing JSON files in hidden directories, you can use these commands to synchronize your LLM profiles from `llm.yaml` directly into Cline's settings.

## Overview

Cline is an AI-powered software engineering agent that integrates directly into VS Code. To maximize its effectiveness, it requires precise configuration of the LLMs used for different stages of its workflow.

### The "Plan" vs "Act" Paradigm
Cline operates in two distinct modes:
- **Plan Mode**: Used for analyzing requirements, architecting solutions, and creating a roadmap. This typically requires a model with strong reasoning and high-context windows.
- **Act Mode**: Used for implementing the plan, writing code, and executing CLI commands. This requires a model that is precise, follows instructions strictly, and is efficient.

Cline stores its configuration in two primary files located in `~/.cline/data`:
- `globalState.json`: Contains general settings, including the active model, API base URLs, and extension state.
- `secrets.json`: Stores sensitive information like API keys.

This command suite allows you to manage both files safely, with built-in backup mechanisms and profile resolution.

## Setup & Configuration

### 1. Profile-based Model Synchronization

The most powerful feature is the ability to set Cline's "Plan" and "Act" models based on a profile defined in your global `llm.yaml` configuration.

```bash
cmc llm cline <profile> [--plan <model>] [--act <model>]
```

#### Parameters:
- `<profile>`: The name of the profile in `llm.yaml` (e.g., `uva.gemma`). The command extracts the `model` and `apiBase` from this profile.
- `--plan <model>`: (Optional) Overrides the model used for the "Plan" mode.
- `--act <model>`: (Optional) Overrides the model used for the "Act" mode.

#### Logic:
- If neither `--plan` nor `--act` is provided, both are set to the model defined in the specified profile.
- The command identifies the `openAiBaseUrl` from the profile's `apiBase`.
- **Interactive Confirmation**: Before any changes are applied, the command prints a "Proposed Changes" summary (e.g., `planModeOpenAiModelId: old-model` $\rightarrow$ `new-model`). You must confirm with `y` to apply the changes.

#### Safety & Backups:
Before modifying `globalState.json`, the command creates an incremental backup using the `backup_name` utility. 
- Example backup files: `globalState.json.bak.1`, `globalState.json.bak.2`, etc.
- This ensures you can always revert to a previous state if a configuration change causes issues.

#### Example:
Set both plan and act models to the `uva.gemma` profile:
```bash
cmc llm cline uva.gemma
```

Set a specific model for acting while using the `uva.gemma` profile for planning:
```bash
cmc llm cline uva.gemma --act gemma-2-9b
```

**Pro Tip:** For the best experience, use a larger model (like `uva.gemma2` or a 27B+ model) for `--plan` and a faster, more concise model for `--act`.

### Command Summary Table

| Command / Option | Description | Example |
| :--- | :--- | :--- |
| `<profile>` | Sync plan/act models from `llm.yaml` profile | `cmc llm cline uva.gemma` |
| `--plan <model>` | Override the planning model | `--plan gemma-2-27b` |
| `--act <model>` | Override the acting model | `--act gemma-2-9b` |
| `list` | List all values in `globalState.json` | `cmc llm cline list` |
| `probe` | Show key values set via the GUI | `cmc llm cline probe` |
| `get <key>` | Retrieve a specific config value | `cmc llm cline get planMode...` |
| `set <key> <val>` | Update a specific config value | `cmc llm cline set theme dark` |
| `edit` | Open `globalState.json` in default editor | `cmc llm cline edit` |
| `secrets set` | Securely save a value in `secrets.json` | `cmc llm cline secrets set api_key ...` |
| `secrets get` | Retrieve a value from `secrets.json` | `cmc llm cline secrets get api_key` |

---

### 2. General Configuration Management

You can perform granular operations on the `globalState.json` file.

#### List All Configurations
View all current settings in `globalState.json`.
```bash
cmc llm cline list
```

#### Retrieve a Specific Value
Get the value of a specific key from the configuration.
```bash
cmc llm cline get <key>
```
*Example:* `cmc llm cline get planModeOpenAiModelId`

#### Set a Configuration Value
Update or create a setting in the global state.
```bash
cmc llm cline set <key> <value>
```
*Example:* `cmc llm cline set someSetting true`

#### Edit Configuration File
Open the `globalState.json` file directly in your system's default text editor for complex edits.
```bash
cmc llm cline edit
```

---

### 3. Secrets Management

Manage sensitive API keys and tokens stored in `secrets.json`.

#### Set a Secret
Securely add or update a secret.
```bash
cmc llm cline secrets set <key> <value>
```

#### Get a Secret
Retrieve the value of a specific secret.
```bash
cmc llm cline secrets get <key>
```

## Technical Details

### Integration with `llm.yaml`
The command leverages the `VLLMConfig` class to resolve profiles. It looks for the profile in the standard Cloudmesh LLM configuration path (usually `~/.config/cloudmesh/llm.yaml`).

### File Paths
- **Config Directory**: `~/.cline/data`
- **Global State**: `~/.cline/data/globalState.json`
- **Secrets**: `~/.cline/data/secrets.json`