I'll create a detailed implementation plan for the `cmc env` command suite with a dedicated `EnvManager` class. Let me first check the existing CLI structure to understand the framework.

## Implementation Plan: `cmc env` Command Suite

### 1. Directory Structure

```
src/cloudmesh/ai/vllm/
├── __init__.py
├── config.py           # Existing - has EnvManager integrated
├── docker_manager.py   # Existing - docker env support
├── server.py           # Existing - remote upload methods
├── env_manager.py      # NEW - Core EnvManager class
└── ...

src/cloudmesh/ai/command/
├── vllm.py             # Existing main command entry
├── env.py              # NEW - cmc env subcommands
```

### 2. Core `EnvManager` Class Design

```python
# src/cloudmesh/ai/vllm/env_manager.py

class EnvManager:
    """Manages environment variables across local and remote systems."""
    
    DEFAULT_LOCAL_PATH = ".env"
    DEFAULT_REMOTE_PATH = "~/.config/cloudmesh/.env"
    DEFAULT_REMOTE_DIR = "~/.config/cloudmesh"
    
    def __init__(self, host: str = None, local_path: str = None, remote_path: str = None):
        self.host = host  # None = local only
        self.local_path = local_path or self.DEFAULT_LOCAL_PATH
        self.remote_path = remote_path or self.DEFAULT_REMOTE_PATH
    
    # === Core Operations ===
    
    def probe(self) -> dict:
        """
        Compare local and remote .env files.
        Returns dict with:
        - local_vars: parsed local env vars
        - remote_vars: parsed remote env vars  
        - diff: differences between them
        - local_perms: local file permissions
        - remote_perms: remote file permissions
        - status: 'identical', 'local_newer', 'remote_newer', 'missing_local', 'missing_remote'
        """
        pass
    
    def sync(self, merge_strategy: str = "merge_local_wins") -> bool:
        """
        Sync local env to remote with intelligent merging.
        
        Strategies:
        - 'local_wins': Local values override remote
        - 'remote_wins': Remote values override local  
        - 'merge': Keep both, local wins on conflict
        - 'ask': Interactive prompt for conflicts
        
        Returns True if sync successful.
        """
        pass
    
    def copy(self, force: bool = False) -> bool:
        """
        Copy local env to remote (destructive - replaces remote).
        
        Args:
            force: Overwrite without prompting
        
        Returns True if copy successful.
        """
        pass
    
    # === Helper Methods ===
    
    def _parse_env_file(self, content: str) -> dict:
        """Parse .env file content into key-value dict."""
        pass
    
    def _render_env_file(self, vars: dict, comments: dict = None) -> str:
        """Render env vars back to .env format with optional comments."""
        pass
    
    def _calculate_diff(self, local: dict, remote: dict) -> dict:
        """
        Calculate diff between env var sets.
        Returns: {
            'added': [...],      # In local, not in remote
            'removed': [...],    # In remote, not in local  
            'changed': [...],    # Different values
            'unchanged': [...]   # Same values
        }
        """
        pass
    
    def _check_permissions(self, path: str, is_remote: bool = False) -> dict:
        """
        Check file permissions.
        Returns: {
            'exists': bool,
            'mode': '644' etc,
            'owner': 'user',
            'group': 'group',
            'secure': bool  # True if mode <= 644
        }
        """
        pass
    
    def _ssh_execute(self, cmd: str) -> tuple:
        """Execute command on remote host via SSH. Returns (stdout, stderr, returncode)."""
        pass
    
    def _ssh_upload(self, content: str, remote_path: str) -> bool:
        """Upload content to remote path via SSH/SCP."""
        pass
    
    def _ssh_download(self, remote_path: str) -> str:
        """Download content from remote path via SSH."""
        pass
```

### 3. CLI Commands (`src/cloudmesh/ai/command/env.py`)

```python
import click
from cloudmesh.ai.vllm.env_manager import EnvManager
from cloudmesh.ai.common.io import console

@click.group()
def env_group():
    """Environment variable management commands."""
    pass

@env_group.command()
@click.argument('host')
@click.option('--local', '-l', default='.env', help='Local env file path')
@click.option('--remote', '-r', default=None, help='Remote env file path')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'yaml']), default='table')
def probe(host, local, remote, output_format):
    """
    Compare local .env with HOST environment.
    
    Shows differences, file permissions, and security status.
    
    Example:
        cmc env probe uva
        cmc env probe dgx --local .env.production --format json
    """
    manager = EnvManager(host=host, local_path=local, remote_path=remote)
    result = manager.probe()
    
    # Output formatting logic
    if output_format == 'table':
        _print_probe_table(result)
    elif output_format == 'json':
        console.print_json(result)
    else:
        import yaml
        console.print(yaml.safe_dump(result))

@env_group.command()
@click.argument('host')
@click.option('--local', '-l', default='.env', help='Local env file path')
@click.option('--remote', '-r', default=None, help='Remote env file path')
@click.option('--strategy', '-s', 
              type=click.Choice(['local_wins', 'remote_wins', 'ask']), 
              default='local_wins',
              help='Conflict resolution strategy')
@click.option('--dry-run', is_flag=True, help='Show what would be changed without syncing')
def sync(host, local, remote, strategy, dry_run):
    """
    Merge local .env into HOST env file intelligently.
    
    Merges environment variables rather than replacing.
    
    Example:
        cmc env sync uva
        cmc env sync uva --strategy ask  # Interactive prompt for conflicts
        cmc env sync dgx --dry-run       # Preview changes
    """
    manager = EnvManager(host=host, local_path=local, remote_path=remote)
    
    if dry_run:
        result = manager.probe()
        _print_sync_preview(result, strategy)
        return
    
    success = manager.sync(merge_strategy=strategy)
    if success:
        console.ok(f"Successfully synced env to {host}:{manager.remote_path}")
    else:
        console.error("Sync failed")

@env_group.command()
@click.argument('host')
@click.option('--local', '-l', default='.env', help='Local env file path')
@click.option('--remote', '-r', default=None, help='Remote env file path')
@click.option('--force', '-f', is_flag=True, help='Overwrite without confirmation')
@click.option('--backup', '-b', is_flag=True, help='Create backup of remote file')
def cp(host, local, remote, force, backup):
    """
    Copy local .env to HOST (replaces remote file).
    
    WARNING: This is destructive! Use 'sync' for merging.
    
    Example:
        cmc env cp uva
        cmc env cp dgx --force --backup
    """
    manager = EnvManager(host=host, local_path=local, remote_path=remote)
    
    if not force:
        if not console.ynchoice(f"Replace remote env on {host}?", default=False):
            console.print("Cancelled")
            return
    
    if backup:
        manager._backup_remote()
    
    success = manager.copy(force=force)
    if success:
        console.ok(f"Copied env to {host}:{manager.remote_path}")
    else:
        console.error("Copy failed")

# Additional utility commands
@env_group.command()
@click.option('--local', '-l', default='.env')
@click.option('--output', '-o', default=None)
def secure(local, output):
    """
    Secure local .env file (permissions 600).
    
    Also validates for common security issues.
    """
    pass

@env_group.command()
@click.argument('host')
@click.option('--remote', '-r', default=None)
def edit(host, remote):
    """
    Edit remote .env file using local editor.
    
    Downloads, opens editor, uploads on save.
    """
    pass

def _print_probe_table(result):
    """Pretty print probe results in table format."""
    pass

def _print_sync_preview(result, strategy):
    """Show preview of sync operation."""
    pass
```

### 4. Integration with Main CLI

```python
# In src/cloudmesh/ai/command/vllm.py or appropriate entry point

from cloudmesh.ai.command.env import env_group

# Add as subcommand
cli.add_command(env_group, name='env')
```

### 5. Smart Sync Algorithm

```python
def sync(self, merge_strategy='local_wins'):
    """
    Intelligent 3-way merge:
    
    1. Load local env vars + comments
    2. Load remote env vars + comments  
    3. Calculate diff (added, removed, changed)
    4. Apply merge strategy:
       - 'local_wins': Use local values for all changes
       - 'remote_wins': Keep remote values where they exist
       - 'ask': Prompt for each conflict
    5. Preserve comments from winning side
    6. Write merged result to remote
    7. Set secure permissions (600)
    """
    local_vars, local_comments = self._load_env_with_comments(self.local_path)
    
    try:
        remote_content = self._ssh_download(self.remote_path)
        remote_vars, remote_comments = self._parse_env_with_comments(remote_content)
    except FileNotFoundError:
        remote_vars, remote_comments = {}, {}
    
    diff = self._calculate_diff(local_vars, remote_vars)
    
    merged = {}
    merged_comments = {}
    
    # Handle different cases
    for key in diff['unchanged']:
        merged[key] = local_vars[key]  # Same value either side
        merged_comments[key] = local_comments.get(key, remote_comments.get(key))
    
    for key in diff['added']:
        # Only in local
        merged[key] = local_vars[key]
        merged_comments[key] = local_comments.get(key)
    
    for key in diff['removed']:
        if merge_strategy == 'local_wins':
            # Don't include - was removed locally
            pass
        else:
            # Keep remote value
            merged[key] = remote_vars[key]
            merged_comments[key] = remote_comments.get(key)
    
    for key in diff['changed']:
        if merge_strategy == 'local_wins':
            merged[key] = local_vars[key]
            merged_comments[key] = local_comments.get(key)
        elif merge_strategy == 'remote_wins':
            merged[key] = remote_vars[key]
            merged_comments[key] = remote_comments.get(key)
        else:  # ask
            # Interactive prompt
            choice = self._prompt_merge_choice(key, local_vars[key], remote_vars[key])
            if choice == 'local':
                merged[key] = local_vars[key]
                merged_comments[key] = local_comments.get(key)
            else:
                merged[key] = remote_vars[key]
                merged_comments[key] = remote_comments.get(key)
    
    # Render and upload
    content = self._render_env_file(merged, merged_comments)
    success = self._ssh_upload(content, self.remote_path)
    
    if success:
        self._ssh_execute(f"chmod 600 {self.remote_path}")
    
    return success
```

### 6. Probe Output Example

```
$ cmc env probe uva

╭──────────────────────────────────────────────────────────────╮
│  Environment Comparison: .env ↔ uva:~/.config/cloudmesh/.env  │
╰──────────────────────────────────────────────────────────────╯

Local File (.env):
  Path: /home/user/project/.env
  Permissions: 644 (⚠️  WARNING: Should be 600)
  Size: 1.2 KB
  Last Modified: 2024-01-15 14:30:22

Remote File (uva:~/.config/cloudmesh/.env):
  Path: /home/user/.config/cloudmesh/.env
  Permissions: 600 ✅
  Size: 0.9 KB
  Last Modified: 2024-01-14 09:15:00

Comparison:
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Variable            │ Local               │ Remote              │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ VLLM_API_KEY        │ ***                 │ ***                 │
│ VLLM_MODEL          │ gemma-2            │ gemma               │  ⚠️  CHANGED
│ VLLM_PORT           │ 8000                │ 8000                │  ✅
│ CUDA_VISIBLE_DEVICES│ 0,1                 │ (not set)           │  📝 ADDED
│ HF_TOKEN            │ (not set)           │ ***                 │  🗑️  REMOVED
└─────────────────────┴─────────────────────┴─────────────────────┘

Summary:
  3 unchanged | 1 changed | 1 added | 1 removed
  Status: Local is newer
```

### 7. Key Implementation Details

| Feature | Implementation |
|---------|---------------|
| **SSH Connection** | Reuse existing SSH config from `~/.ssh/config` |
| **File Transfer** | Try `scp` first, fallback to `ssh cat << 'EOF'` |
| **Security** | Auto-set permissions to 600 on remote |
| **Backup** | Create `.env.backup.YYYYMMDD_HHMMSS` before changes |
| **Comments** | Preserve and merge comments intelligently |
| **Validation** | Warn on secrets in env, insecure permissions |
| **Diff Display** | Mask secret values (show `***` for API keys) |

### Phase 1: Core Class (est. 2-3 hours)
- [ ] Create `EnvManager` class with all methods
- [ ] Implement `probe()`, `sync()`, `copy()`
- [ ] Add helper methods for parsing/rendering

### Phase 2: CLI Commands (est. 1-2 hours)
- [ ] Create `env.py` command module
- [ ] Implement `probe`, `sync`, `cp` commands
- [ ] Add output formatting (table, json, yaml)

### Phase 3: Integration & Testing (est. 1 hour)
- [ ] Wire up to main CLI
- [ ] Test local operations
- [ ] Test SSH operations
- [ ] Create documentation

**Total: 4-6 hours of implementation work**

Should I proceed with implementation?