I also want that we have a command that can set the plan and act based on --plan --act and if not specified set both to the same. from cmc llm cline uva.gemma whene uva.gemma is an example from the llm.yaml file 
it will print fisrt what is proposed to be changed. Then we ask if thats ok and than change it in the file for cline. before it is chnaged a backup file is created with incremental backup.

use for that `backup_name` in `cloudmesh-ai-common/src/cloudmesh/ai/common/util.py` that does exactly this: it generates a filename with an incremental `.bak.N` suffix.



Understood. I will implement the command as `cmc llm cline` 

### Updated Command Structure:
- `cmc llm cline list`: List all current global configurations.
- `cmc llm cline get <key>`: Retrieve a specific configuration value.
- `cmc llm cline set <key> <value>`: Update or create a configuration setting.
- `cmc llm cline edit`: Open the global configuration file in your default editor.
- `cmc llm cline secrets set <key> <value>`: Securely set a secret.
- `cmc llm cline secrets get <key>`: Retrieve a specific secret.

### Updated Technical Implementation:
1. **Core Logic**: Implement `ClineManager` in `src/cloudmesh/ai/vllm/cline_manager.py` to handle `~/.cline/data` JSON files.
2. **CLI Layer**: Implement the `cline` command group in `src/cloudmesh/ai/command/cline.py` and integrate it as a subcommand into the existing `llm` group in `src/cloudmesh/ai/command/vllm.py`.
3. **Verification**: Ensure that `cmc llm cline` is correctly routed via the `llm` entry point.

If this looks correct, please **toggle to Act mode** and I will start the implementation.

### Final Implementation Plan:

**1. Command: `cmc llm cline <profile> [--plan <model>] [--act <model>]`**
- **Profile Lookup**: It will look up the specified profile (e.g., `uva.gemma`) in `llm.yaml`. It will extract the model name and the server host/port to determine the `openAiBaseUrl`.
- **Logic**:
    - If `--plan` is provided, it overrides the plan model.
    - If `--act` is provided, it overrides the act model.
    - If neither is provided, both are set to the model from the profile.
- **Interactive Flow**:
    - The command will print the proposed changes (e.g., `planModeOpenAiModelId: model-a` $\rightarrow$ `model-b`).
    - It will ask for confirmation: `Is this ok? (y/n)`.
- **Safety**: Before applying changes, it will create an incremental backup of `globalState.json` (e.g., `globalState.json.bak.1`) using the `backup_name` utility from `cloudmesh-ai-common`.

**2. Command: `cmc llm cline [list|get <key>|set <key> <value>|edit]`**
- I will also implement the general management commands discussed previously for direct access to `globalState.json` and `secrets.json`.

**3. Technical Details**:
- **Core**: `src/cloudmesh/ai/vllm/cline_manager.py` will handle the JSON manipulation and the integration with `llm.yaml`.
- **CLI**: `src/cloudmesh/ai/command/cline.py` will define the CLI interface.
- **Integration**: Integrated as a subcommand of `llm` in `src/cloudmesh/ai/command/vllm.py`.

If this is correct, please **toggle to Act mode** and I will implement it.
