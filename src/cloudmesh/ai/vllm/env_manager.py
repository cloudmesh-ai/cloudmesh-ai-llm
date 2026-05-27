"""
Environment Manager for cloudmesh-ai-llm.

Manages .env files across local and remote systems with intelligent
syncing, probing, and copying capabilities.
"""

import os
import re
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from cloudmesh.ai.common.io import console


class EnvManager:
    """
    Manages environment variables across local and remote systems.
    
    Provides intelligent syncing, probing, and secure file handling
    for .env files with support for SSH-based remote operations.
    """
    
    DEFAULT_LOCAL_PATH = ".env"
    DEFAULT_REMOTE_PATH = "~/.config/cloudmesh/.env"
    DEFAULT_REMOTE_DIR = "~/.config/cloudmesh"
    
    # Patterns for masking secrets
    SECRET_PATTERNS = [
        r'.*API_KEY.*',
        r'.*SECRET.*',
        r'.*PASSWORD.*',
        r'.*TOKEN.*',
        r'.*PRIVATE.*',
        r'.*CREDENTIAL.*',
        r'.*AUTH.*',
        r'.*_KEY.*',
    ]
    
    def __init__(
        self,
        host: Optional[str] = None,
        local_path: Optional[str] = None,
        remote_path: Optional[str] = None
    ):
        """
        Initialize EnvManager.
        
        Args:
            host: Remote hostname (None for local-only operations)
            local_path: Path to local .env file (if None, searches default locations)
            remote_path: Path to remote .env file
        """
        self.host = host
        
        # Determine if we're working remotely
        self.is_remote = host is not None
        
        # Resolve local path - search multiple locations if not explicitly specified
        if local_path:
            self.local_path = local_path
        else:
            self.local_path = self._resolve_local_env_path()
        
        self.remote_path = remote_path or self.DEFAULT_REMOTE_PATH
    
    def _resolve_local_env_path(self) -> str:
        """
        Resolve the local env file path by searching multiple locations.
        
        Searches in order:
        1. .env in current directory
        2. ~/.config/cloudmesh/.env
        
        Returns:
            Path to the found env file, or DEFAULT_LOCAL_PATH if not found
        """
        search_paths = [
            self.DEFAULT_LOCAL_PATH,  # .env in current directory
            os.path.expanduser(self.DEFAULT_REMOTE_PATH),  # ~/.config/cloudmesh/.env
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # Return default if not found - let callers handle the error
        return self.DEFAULT_LOCAL_PATH
        
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def probe(self) -> Dict:
        """
        Compare local and remote .env files.
        
        Returns comprehensive comparison including:
        - local_vars: parsed local env vars
        - remote_vars: parsed remote env vars
        - diff: differences between them
        - local_perms: local file permissions
        - remote_perms: remote file permissions
        - status: comparison status
        
        Returns:
            Dict with probe results
        """
        result = {
            'host': self.host,
            'local_path': os.path.abspath(self.local_path) if os.path.exists(self.local_path) else self.local_path,
            'remote_path': self.remote_path,
            'local_exists': False,
            'remote_exists': False,
            'local_vars': {},
            'remote_vars': {},
            'local_comments': {},
            'remote_comments': {},
            'local_perms': None,
            'remote_perms': None,
            'local_size': None,
            'remote_size': None,
            'local_mtime': None,
            'remote_mtime': None,
            'diff': {},
            'status': 'unknown',
            'timestamp': datetime.now().isoformat()
        }
        
        # Probe local file
        if os.path.exists(self.local_path):
            result['local_exists'] = True
            result['local_perms'] = self._check_local_permissions(self.local_path)
            result['local_size'] = os.path.getsize(self.local_path)
            result['local_mtime'] = datetime.fromtimestamp(os.path.getmtime(self.local_path)).isoformat()
            local_content = self._read_local_file(self.local_path)
            result['local_vars'], result['local_comments'] = self._parse_env_with_comments(local_content)
        
        # Probe remote file (if applicable)
        if self.is_remote:
            try:
                remote_content = self._ssh_download(self.remote_path)
                result['remote_exists'] = True
                result['remote_perms'] = self._check_remote_permissions(self.remote_path)
                result['remote_size'] = len(remote_content.encode('utf-8'))
                # Get mtime from remote
                mtime_result = self._ssh_execute(f"stat -c '%Y' {self.remote_path} 2>/dev/null || stat -f '%m' {self.remote_path}")
                if mtime_result[2] == 0:
                    try:
                        mtime = int(mtime_result[0].strip())
                        result['remote_mtime'] = datetime.fromtimestamp(mtime).isoformat()
                    except (ValueError, TypeError):
                        pass
                result['remote_vars'], result['remote_comments'] = self._parse_env_with_comments(remote_content)
            except FileNotFoundError:
                result['remote_exists'] = False
                result['status'] = 'missing_remote'
            except Exception as e:
                result['remote_error'] = str(e)
                result['status'] = 'remote_error'
        
        # Calculate diff if both exist
        if result['local_exists'] and (not self.is_remote or result['remote_exists']):
            if self.is_remote:
                result['diff'] = self._calculate_diff(
                    result['local_vars'],
                    result['remote_vars']
                )
                result['status'] = self._determine_status(result)
            else:
                result['status'] = 'local_only'
        elif not result['local_exists'] and result.get('remote_exists'):
            result['status'] = 'missing_local'
        
        return result
    
    def sync(self, merge_strategy: str = "local_wins", dry_run: bool = False) -> bool:
        """
        Sync local env to remote with intelligent merging.
        
        Args:
            merge_strategy: How to handle conflicts ('local_wins', 'remote_wins', 'ask')
            dry_run: If True, only show what would be changed
        
        Returns:
            True if sync successful
        """
        if not self.is_remote:
            console.error("Sync requires a remote host")
            return False
        
        if not os.path.exists(self.local_path):
            console.error(f"Local env file not found: {self.local_path}")
            return False
        
        # Load local
        local_content = self._read_local_file(self.local_path)
        local_vars, local_comments = self._parse_env_with_comments(local_content)
        
        # Load remote (if exists)
        try:
            remote_content = self._ssh_download(self.remote_path)
            remote_vars, remote_comments = self._parse_env_with_comments(remote_content)
            remote_exists = True
        except FileNotFoundError:
            remote_vars, remote_comments = {}, {}
            remote_exists = False
        
        # Calculate diff
        diff = self._calculate_diff(local_vars, remote_vars)
        
        if dry_run:
            self._print_sync_preview(local_vars, remote_vars, diff, merge_strategy)
            return True
        
        # Merge
        merged_vars, merged_comments = self._merge_env_vars(
            local_vars, remote_vars,
            local_comments, remote_comments,
            diff, merge_strategy
        )
        
        # Render merged content
        merged_content = self._render_env_file(merged_vars, merged_comments)
        
        # Backup remote if it exists
        if remote_exists:
            self._backup_remote()
        
        # Upload
        success = self._ssh_upload(merged_content, self.remote_path)
        
        if success:
            # Set secure permissions
            self._ssh_execute(f"chmod 600 {self.remote_path}")
            console.ok(f"Synced env to {self.host}:{self.remote_path}")
        
        return success
    
    def copy(self, force: bool = False, backup: bool = True) -> bool:
        """
        Copy local env to remote (destructive - replaces remote).
        
        Args:
            force: Overwrite without prompting
            backup: Create backup of remote file
        
        Returns:
            True if copy successful
        """
        if not self.is_remote:
            # Local copy
            try:
                shutil.copy2(self.local_path, self.remote_path)
                os.chmod(self.remote_path, 0o600)
                console.ok(f"Copied to {self.remote_path}")
                return True
            except Exception as e:
                console.error(f"Copy failed: {e}")
                return False
        
        # Remote copy
        if not os.path.exists(self.local_path):
            console.error(f"Local env file not found: {self.local_path}")
            return False
        
        # Check if remote exists
        remote_exists = self._remote_file_exists(self.remote_path)
        
        if remote_exists and not force:
            if not console.ynchoice(
                f"Remote file exists at {self.remote_path}. Overwrite?",
                default=False
            ):
                console.print("Copy cancelled")
                return False
        
        # Backup if requested
        if backup and remote_exists:
            self._backup_remote()
        
        # Ensure remote directory exists
        remote_dir = os.path.dirname(self.remote_path)
        self._ssh_execute(f"mkdir -p {remote_dir}")
        
        # Upload
        success = self._scp_upload(self.local_path, self.remote_path)
        
        if success:
            # Set secure permissions
            self._ssh_execute(f"chmod 600 {self.remote_path}")
            console.ok(f"Copied env to {self.host}:{self.remote_path}")
        
        return success
    
    # =========================================================================
    # Additional Utility Methods
    # =========================================================================
    
    def secure(self, path: Optional[str] = None) -> bool:
        """
        Secure env file permissions (600).
        
        Args:
            path: Path to file (default: self.local_path)
        
        Returns:
            True if successful
        """
        target_path = path or self.local_path
        
        try:
            if self.is_remote and path == self.remote_path:
                self._ssh_execute(f"chmod 600 {target_path}")
            else:
                os.chmod(target_path, 0o600)
            console.ok(f"Set secure permissions (600) on {target_path}")
            return True
        except Exception as e:
            console.error(f"Failed to secure {target_path}: {e}")
            return False
    
    def validate(self, path: Optional[str] = None) -> Dict:
        """
        Validate env file for security issues.
        
        Args:
            path: Path to file (default: self.local_path)
        
        Returns:
            Dict with validation results
        """
        target_path = path or self.local_path
        is_remote = self.is_remote and (path == self.remote_path or not path)
        
        result = {
            'path': target_path,
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check permissions
        if is_remote:
            perms = self._check_remote_permissions(target_path)
        else:
            perms = self._check_local_permissions(target_path)
        
        if perms and perms['mode_int'] > 0o600:
            result['warnings'].append(
                f"Insecure permissions: {perms['mode']} (should be 600)"
            )
        
        # Load and check content
        try:
            if is_remote:
                content = self._ssh_download(target_path)
            else:
                content = self._read_local_file(target_path)
            vars_dict, _ = self._parse_env_with_comments(content)
            
            # Check for empty values on critical vars
            critical_vars = ['VLLM_API_KEY', 'CLOUDMESH_AI_API_KEY', 'HF_TOKEN']
            for var in critical_vars:
                if var in vars_dict and not vars_dict[var]:
                    result['warnings'].append(f"{var} is set but empty")
            
            # Check for secrets in values that look like URLs with credentials
            for key, value in vars_dict.items():
                if '://' in str(value) and ('@' in str(value) or ':' in str(value)):
                    if any(pattern in key.upper() for pattern in ['URL', 'URI', 'ENDPOINT']):
                        result['warnings'].append(
                            f"{key} may contain credentials in URL - consider using separate env vars"
                        )
                        
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f"Failed to parse file: {e}")
        
        return result
    
    def edit_remote(self, local_path: Optional[str] = None) -> bool:
        """
        Edit remote .env file using local editor.
        
        Downloads, opens editor, uploads on save.
        If remote file doesn't exist, offers to create or copy from local.
        
        Args:
            local_path: Optional local .env path to copy from
        
        Returns:
            True if successful
        """
        if not self.is_remote:
            console.error("edit_remote requires a remote host")
            return False
        
        try:
            # Download remote content
            try:
                remote_content = self._ssh_download(self.remote_path)
                remote_exists = True
            except FileNotFoundError:
                remote_exists = False
                remote_content = ""
            
            # Handle missing remote file
            if not remote_exists:
                console.warning(f"Remote file not found: {self.remote_path}")
                console.print(f"\n[dim]Source (local): {local_path if local_path else self.local_path}[/dim]")
                console.print(f"[dim]Destination (remote): {self.host}:{self.remote_path}[/dim]")
                console.print("\n[bold]Options:[/bold]")
                console.print("  1. Create new empty file")
                if local_path and os.path.exists(local_path):
                    console.print(f"  2. Copy from local: {local_path} → {self.host}:{self.remote_path}")
                elif os.path.exists(self.local_path):
                    console.print(f"  2. Copy from local: {self.local_path} → {self.host}:{self.remote_path}")
                console.print("  3. Cancel")
                
                choice = console.input("\nSelect option [1-3]: ").strip()
                
                if choice == '2' and local_path and os.path.exists(local_path):
                    # Copy from local
                    console.print(f"[blue]Copying from {local_path} to {self.host}:{self.remote_path}...[/blue]")
                    remote_content = self._read_local_file(local_path)
                    # Ensure directory exists
                    remote_dir = os.path.dirname(self.remote_path)
                    self._ssh_execute(f"mkdir -p {remote_dir}")
                    # Upload initial content
                    self._ssh_upload(remote_content, self.remote_path)
                    self._ssh_execute(f"chmod 600 {self.remote_path}")
                    console.ok(f"Copied local .env to {self.host}:{self.remote_path}")
                elif choice == '1':
                    # Create new empty file
                    remote_content = "# New environment file\n"
                    # Ensure directory exists
                    remote_dir = os.path.dirname(self.remote_path)
                    self._ssh_execute(f"mkdir -p {remote_dir}")
                    # Create initial file
                    self._ssh_upload(remote_content, self.remote_path)
                    self._ssh_execute(f"chmod 600 {self.remote_path}")
                    console.ok(f"Created new file at {self.host}:{self.remote_path}")
                else:
                    console.print("Cancelled")
                    return False
            
            # Create temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.env',
                delete=False
            ) as tmp:
                tmp.write(remote_content)
                tmp_path = tmp.name
            
            # Count initial lines/vars for summary
            initial_vars = self._parse_env_with_comments(remote_content)[0]
            
            # Get editor from env
            editor = os.environ.get('EDITOR', 'nano')
            
            # Open editor
            console.print(f"\n[blue]Opening editor: {editor}[/blue]")
            console.print(f"[dim]File: {self.remote_path} on {self.host}[/dim]\n")
            result = subprocess.run([editor, tmp_path])
            
            if result.returncode != 0:
                console.error("Editor exited with error")
                os.unlink(tmp_path)
                return False
            
            # Read modified content
            with open(tmp_path, 'r') as f:
                new_content = f.read()
            
            # Upload if changed
            if new_content != remote_content:
                # Show diff summary
                new_vars = self._parse_env_with_comments(new_content)[0]
                added = set(new_vars.keys()) - set(initial_vars.keys())
                removed = set(initial_vars.keys()) - set(new_vars.keys())
                
                self._backup_remote()
                self._ssh_upload(new_content, self.remote_path)
                self._ssh_execute(f"chmod 600 {self.remote_path}")
                
                console.ok("Remote .env updated")
                if added:
                    console.print(f"[green]  + Added: {', '.join(sorted(added))}[/green]")
                if removed:
                    console.print(f"[red]  - Removed: {', '.join(sorted(removed))}[/red]")
            else:
                console.print("No changes made")
            
            # Cleanup
            os.unlink(tmp_path)
            return True
            
        except Exception as e:
            console.error(f"Edit failed: {e}")
            return False
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _parse_env_with_comments(self, content: str) -> Tuple[Dict, Dict]:
        """
        Parse .env content into vars and comments.
        
        Args:
            content: .env file content
        
        Returns:
            Tuple of (vars_dict, comments_dict)
        """
        vars_dict = {}
        comments = {}
        current_comment = []
        
        for line_num, line in enumerate(content.split('\n'), 1):
            stripped = line.strip()
            
            # Handle comments
            if stripped.startswith('#'):
                current_comment.append(stripped[1:].strip())
                continue
            
            # Handle empty lines
            if not stripped:
                current_comment = []
                continue
            
            # Parse key=value
            if '=' in stripped:
                key, value = stripped.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                vars_dict[key] = value
                
                if current_comment:
                    comments[key] = '\n'.join(current_comment)
                    current_comment = []
        
        return vars_dict, comments
    
    def _render_env_file(self, vars_dict: Dict, comments_dict: Dict = None) -> str:
        """
        Render env vars back to .env format.
        
        Args:
            vars_dict: Dictionary of env vars
            comments_dict: Dictionary of comments for each key
        
        Returns:
            .env file content as string
        """
        if comments_dict is None:
            comments_dict = {}
        
        lines = []
        
        # Add header comment
        lines.append(f"# Generated by cloudmesh env manager")
        lines.append(f"# Timestamp: {datetime.now().isoformat()}")
        lines.append("")
        
        # Sort keys for consistent output
        for key in sorted(vars_dict.keys()):
            # Add comment before the key if present
            if key in comments_dict:
                for comment_line in comments_dict[key].split('\n'):
                    lines.append(f"# {comment_line}")
            
            value = vars_dict[key]
            
            # Quote value if it contains special characters
            if ' ' in value or '\n' in value or '"' in value or "'" in value:
                # Use double quotes and escape internal quotes
                escaped = value.replace('"', '\\"')
                lines.append(f'{key}="{escaped}"')
            else:
                lines.append(f"{key}={value}")
        
        return '\n'.join(lines)
    
    def _read_local_file(self, path: str) -> str:
        """Read local file content."""
        with open(path, 'r') as f:
            return f.read()
    
    def _check_local_permissions(self, path: str) -> Optional[Dict]:
        """Check local file permissions."""
        try:
            stat = os.stat(path)
            mode = oct(stat.st_mode)[-3:]
            return {
                'exists': True,
                'mode': mode,
                'mode_int': stat.st_mode & 0o777,
                'owner': stat.st_uid,
                'group': stat.st_gid,
                'secure': (stat.st_mode & 0o777) <= 0o600
            }
        except Exception:
            return None
    
    def _check_remote_permissions(self, path: str) -> Optional[Dict]:
        """Check remote file permissions via SSH."""
        try:
            cmd = f"stat -c '%a %U %G' {path} 2>/dev/null || stat -f '%Lp %Su %Sg' {path}"
            stdout, stderr, rc = self._ssh_execute(cmd)
            if rc != 0:
                return None
            
            parts = stdout.strip().split()
            mode = parts[0] if parts else 'unknown'
            owner = parts[1] if len(parts) > 1 else 'unknown'
            group = parts[2] if len(parts) > 2 else 'unknown'
            
            mode_int = int(mode, 8) if mode.isdigit() else 0o777
            
            return {
                'exists': True,
                'mode': mode,
                'mode_int': mode_int,
                'owner': owner,
                'group': group,
                'secure': mode_int <= 0o600
            }
        except Exception:
            return None
    
    def _calculate_diff(self, local: Dict, remote: Dict) -> Dict:
        """
        Calculate diff between env var sets.
        
        Returns:
            Dict with 'added', 'removed', 'changed', 'unchanged' keys
        """
        local_keys = set(local.keys())
        remote_keys = set(remote.keys())
        
        return {
            'added': sorted(list(local_keys - remote_keys)),
            'removed': sorted(list(remote_keys - local_keys)),
            'changed': sorted([k for k in local_keys & remote_keys if local[k] != remote[k]]),
            'unchanged': sorted([k for k in local_keys & remote_keys if local[k] == remote[k]])
        }
    
    def _determine_status(self, probe_result: Dict) -> str:
        """Determine overall status from probe result."""
        if not probe_result['local_exists'] and not probe_result['remote_exists']:
            return 'both_missing'
        elif not probe_result['local_exists']:
            return 'missing_local'
        elif not probe_result['remote_exists']:
            return 'missing_remote'
        
        # Both exist - compare vars
        diff = probe_result['diff']
        if diff['changed'] or diff['added'] or diff['removed']:
            # Check timestamps if available
            local_mtime = probe_result.get('local_mtime')
            remote_mtime = probe_result.get('remote_mtime')
            if local_mtime and remote_mtime:
                if local_mtime > remote_mtime:
                    return 'local_newer'
                else:
                    return 'remote_newer'
            return 'different'
        
        return 'identical'
    
    def _merge_env_vars(
        self,
        local_vars: Dict, remote_vars: Dict,
        local_comments: Dict, remote_comments: Dict,
        diff: Dict, strategy: str
    ) -> Tuple[Dict, Dict]:
        """
        Merge env vars according to strategy.
        
        Returns:
            Tuple of (merged_vars, merged_comments)
        """
        merged = {}
        merged_comments = {}
        
        # Handle unchanged vars
        for key in diff['unchanged']:
            merged[key] = local_vars[key]
            merged_comments[key] = local_comments.get(key, remote_comments.get(key, ''))
        
        # Handle added (local-only)
        for key in diff['added']:
            merged[key] = local_vars[key]
            merged_comments[key] = local_comments.get(key, '')
        
        # Handle removed (remote-only, not in local)
        for key in diff['removed']:
            if strategy == 'remote_wins':
                # Keep remote value
                merged[key] = remote_vars[key]
                merged_comments[key] = remote_comments.get(key, '')
            # else: local_wins - don't include, was removed locally
        
        # Handle changed
        for key in diff['changed']:
            if strategy == 'local_wins':
                merged[key] = local_vars[key]
                merged_comments[key] = local_comments.get(key, '')
            elif strategy == 'remote_wins':
                merged[key] = remote_vars[key]
                merged_comments[key] = remote_comments.get(key, '')
            elif strategy == 'ask':
                choice = self._prompt_merge_choice(
                    key, local_vars[key], remote_vars[key]
                )
                if choice == 'local':
                    merged[key] = local_vars[key]
                    merged_comments[key] = local_comments.get(key, '')
                else:
                    merged[key] = remote_vars[key]
                    merged_comments[key] = remote_comments.get(key, '')
        
        return merged, merged_comments
    
    def _prompt_merge_choice(self, key: str, local_val: str, remote_val: str) -> str:
        """Interactive prompt for merge conflict."""
        # Mask values that look like secrets
        local_display = self._mask_if_secret(key, local_val)
        remote_display = self._mask_if_secret(key, remote_val)
        
        console.print(f"\nConflict for [bold]{key}[/bold]:")
        console.print(f"  [blue]Local:[/blue]  {local_display}")
        console.print(f"  [green]Remote:[/green] {remote_display}")
        
        choice = console.input("Use [l]ocal, [r]emote, or [s]kip? [l/r/s]: ").lower().strip()
        if choice == 'r':
            return 'remote'
        elif choice == 's':
            return 'skip'
        return 'local'  # default
    
    def _mask_if_secret(self, key: str, value: str) -> str:
        """Mask value if key looks like a secret."""
        key_upper = key.upper()
        for pattern in self.SECRET_PATTERNS:
            if re.match(pattern, key_upper):
                if value:
                    return '***'
                return '(empty)'
        # For non-secrets, truncate if very long
        if len(str(value)) > 60:
            return str(value)[:57] + '...'
        return str(value)
    
    def _print_sync_preview(self, local_vars: Dict, remote_vars: Dict, diff: Dict, strategy: str):
        """Print preview of sync operation."""
        show_values = getattr(self, '_show_values', False)
        
        console.print(f"\n[bold]Sync Preview (strategy: {strategy})[/bold]")
        console.print(f"[dim]Source:[/dim] {os.path.abspath(self.local_path) if os.path.exists(self.local_path) else self.local_path}")
        if self.is_remote:
            console.print(f"[dim]Destination:[/dim] {self.host}:{self.remote_path}")
        else:
            console.print(f"[dim]Destination:[/dim] {self.remote_path}")
        console.print("")
        
        if diff['added']:
            console.print("[green]Variables to add:[/green]")
            for key in diff['added']:
                value = local_vars[key]
                display = value if show_values else self._mask_if_secret(key, value)
                console.print(f"  + {key}={display}")
        
        if diff['removed']:
            console.print("[red]Variables to remove:[/red]")
            for key in diff['removed']:
                action = "keep" if strategy == 'remote_wins' else "delete"
                value = remote_vars[key]
                display = value if show_values else self._mask_if_secret(key, value)
                console.print(f"  - {key}={display} ({action})")
        
        if diff['changed']:
            console.print("[yellow]Variables to update:[/yellow]")
            for key in diff['changed']:
                local_val = local_vars[key]
                remote_val = remote_vars[key]
                
                if show_values:
                    local_display = local_val
                    remote_display = remote_val
                else:
                    local_display = self._mask_if_secret(key, local_val)
                    remote_display = self._mask_if_secret(key, remote_val)
                
                if strategy == 'local_wins':
                    console.print(f"  ~ {key}: {remote_display} -> {local_display}")
                elif strategy == 'remote_wins':
                    console.print(f"  ~ {key}: {local_display} -> {remote_display} (remote wins)")
                else:
                    console.print(f"  ? {key}: local={local_display}, remote={remote_display}")
        
        if not any([diff['added'], diff['removed'], diff['changed']]):
            source = os.path.abspath(self.local_path) if os.path.exists(self.local_path) else self.local_path
            dest = f"{self.host}:{self.remote_path}" if self.is_remote else self.remote_path
            console.print(f"[green]No changes needed - files are identical[/green]")
            console.print(f"[dim]{source} → {dest}[/dim]")
        else:
            console.print(f"\nSummary: {len(diff['added'])} to add, {len(diff['removed'])} to remove, {len(diff['changed'])} to update")
    
    def _ssh_execute(self, cmd: str) -> Tuple[str, str, int]:
        """
        Execute command on remote host via SSH.
        
        Returns:
            Tuple of (stdout, stderr, returncode)
        """
        if not self.is_remote:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout, result.stderr, result.returncode
        
        ssh_cmd = ["ssh", self.host, cmd]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    
    def _ssh_download(self, remote_path: str) -> str:
        """Download content from remote path via SSH."""
        if not self.is_remote:
            with open(os.path.expanduser(remote_path), 'r') as f:
                return f.read()
        
        # Check if file exists first
        check_cmd = f"test -f {remote_path} && echo EXISTS || echo MISSING"
        stdout, _, _ = self._ssh_execute(check_cmd)
        
        if 'MISSING' in stdout:
            raise FileNotFoundError(f"Remote file not found: {remote_path}")
        
        # Use cat to get content
        cmd = f"cat {remote_path}"
        stdout, stderr, rc = self._ssh_execute(cmd)
        
        if rc != 0:
            raise RuntimeError(f"Failed to download: {stderr}")
        
        return stdout
    
    def _ssh_upload(self, content: str, remote_path: str) -> bool:
        """Upload content to remote path via SSH/SCP."""
        if not self.is_remote:
            local_path = os.path.expanduser(remote_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'w') as f:
                f.write(content)
            return True
        
        # Method 1: Try SCP first
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            result = subprocess.run(
                ["scp", tmp_path, f"{self.host}:{remote_path}"],
                capture_output=True, text=True
            )
            
            os.unlink(tmp_path)
            
            if result.returncode == 0:
                return True
        except Exception:
            pass
        
        # Method 2: Fallback to SSH with heredoc
        try:
            import shlex
            escaped_content = content.replace("'", "'\\''")
            cmd = f"cat << 'EOF' > {remote_path}\n{escaped_content}\nEOF"
            _, _, rc = self._ssh_execute(cmd)
            return rc == 0
        except Exception as e:
            console.error(f"Upload failed: {e}")
            return False
    
    def _scp_upload(self, local_path: str, remote_path: str) -> bool:
        """Upload file via SCP."""
        try:
            result = subprocess.run(
                ["scp", local_path, f"{self.host}:{remote_path}"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except Exception as e:
            console.error(f"SCP upload failed: {e}")
            # Fallback to content-based upload
            content = self._read_local_file(local_path)
            return self._ssh_upload(content, remote_path)
    
    def _remote_file_exists(self, remote_path: str) -> bool:
        """Check if remote file exists."""
        try:
            cmd = f"test -f {remote_path} && echo YES || echo NO"
            stdout, _, _ = self._ssh_execute(cmd)
            return 'YES' in stdout
        except Exception:
            return False
    
    def _backup_remote(self) -> bool:
        """Create backup of remote file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.remote_path}.backup.{timestamp}"
        
        try:
            self._ssh_execute(f"cp {self.remote_path} {backup_path}")
            console.print(f"[dim]Backup created: {backup_path}[/dim]")
            return True
        except Exception as e:
            console.warning(f"Failed to create backup: {e}")
            return False