"""
Environment variable management commands for cmc env.

Provides commands to probe, sync, copy, and manage .env files
across local and remote systems.
"""

import click
import json
import os
from typing import Optional

from cloudmesh.ai.vllm.env_manager import EnvManager
from cloudmesh.ai.common.io import console


@click.group(name="env")
def env_group():
    """
    Environment variable management commands.
    
    Manage .env files across local and remote systems with
    intelligent syncing, probing, and secure file handling.
    
    Examples:
        cmc env probe uva              # Compare local .env with remote
        cmc env sync uva --dry-run     # Preview sync changes
        cmc env cp uva --backup        # Copy with backup
        cmc env validate               # Check local .env security
    """
    pass


@env_group.command(name="init")
@click.option('--path', '-p', default=None, help='Path to create the .env file (default: .env)')
def init_cmd(path: Optional[str]):
    """
    Initialize a default .env file with a template.
    
    Creates a local .env file containing all available configuration options
    with default values. Use this to quickly start configuring your environment.
    
    Example:
        cmc env init                    # Create default .env
        cmc env init -p .env.local      # Create specific local env
    """
    manager = EnvManager()
    if manager.create_template(path=path):
        console.ok(f"Environment initialized successfully")
    else:
        console.error("Failed to initialize environment")

@env_group.command(name="probe")
@click.argument('host', required=False, default=None)
@click.option('--local', '-l', default=None, help='Local env file path (default: searches .env, then ~/.config/cloudmesh/.env)')
@click.option('--remote', '-r', default=None, help='Remote env file path')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'yaml']), default='table')
def probe_cmd(host: Optional[str], local: Optional[str], remote: Optional[str], output_format: str):
    """
    Compare local .env with HOST environment.
    
    Shows differences, file permissions, and security status.
    If HOST is not specified, only local file info is shown.
    
    Examples:
        cmc env probe              # Show local file info only
        cmc env probe uva          # Compare local with remote host
        cmc env probe dgx --local .env.production --format json
    """
    manager = EnvManager(host=host, local_path=local, remote_path=remote)
    result = manager.probe()
    
    if output_format == 'table':
        _print_probe_table(result, manager)
    elif output_format == 'json':
        console.print(json.dumps(result, indent=2, default=str))
    else:
        try:
            import yaml
            console.print(yaml.safe_dump(result, default_flow_style=False))
        except ImportError:
            console.warning("PyYAML not installed, using JSON output")
            console.print(json.dumps(result, indent=2, default=str))


@env_group.command(name="sync")
@click.argument('host')
@click.option('--local', '-l', default=None, help='Local env file path (default: searches .env, then ~/.config/cloudmesh/.env)')
@click.option('--remote', '-r', default=None, help='Remote env file path')
@click.option('--strategy', '-s', 
              type=click.Choice(['local_wins', 'remote_wins', 'ask']), 
              default='local_wins',
              help='Conflict resolution strategy')
@click.option('--dry-run', '-n', is_flag=True, help='Show what would be changed without syncing')
@click.option('--show', is_flag=True, help='Show actual values (not masked)')
def sync_cmd(host: str, local: Optional[str], remote: Optional[str], strategy: str, dry_run: bool, show: bool):
    """
    Merge local .env into HOST env file intelligently.
    
    Merges environment variables rather than replacing.
    
    Examples:
        cmc env sync uva
        cmc env sync uva --strategy ask  # Interactive prompt for conflicts
        cmc env sync dgx --dry-run       # Preview changes
    """
    manager = EnvManager(host=host, local_path=local, remote_path=remote)
    manager._show_values = show  # Store flag for preview use
    
    if dry_run:
        success = manager.sync(merge_strategy=strategy, dry_run=True)
    else:
        success = manager.sync(merge_strategy=strategy, dry_run=False)
        if success:
            console.ok(f"Successfully synced env to {host}:{manager.remote_path}")
        else:
            console.error("Sync failed")


@env_group.command(name="cp")
@click.argument('host')
@click.option('--local', '-l', default=None, help='Local env file path (default: searches .env, then ~/.config/cloudmesh/.env)')
@click.option('--remote', '-r', default=None, help='Remote env file path')
@click.option('--force', '-f', is_flag=True, help='Overwrite without confirmation')
@click.option('--backup', '-b', is_flag=True, default=True, help='Create backup of remote file')
@click.option('--no-backup', is_flag=True, help='Skip backup creation')
def copy_cmd(host: str, local: Optional[str], remote: Optional[str], force: bool, backup: bool, no_backup: bool):
    """
    Copy local .env to HOST (replaces remote file).
    
    WARNING: This is destructive! Use 'sync' for merging.
    
    Examples:
        cmc env cp uva
        cmc env cp dgx --force --backup
        cmc env cp uva --no-backup
    """
    # Handle --no-backup flag
    create_backup = backup and not no_backup
    
    manager = EnvManager(host=host, local_path=local, remote_path=remote)
    
    success = manager.copy(force=force, backup=create_backup)
    
    if success:
        console.ok(f"Copied env to {host}:{manager.remote_path}")
        if not create_backup:
            console.warning("No backup was created (--no-backup was used)")
    else:
        console.error("Copy failed")


@env_group.command(name="secure")
@click.argument('host', required=False, default=None)
@click.option('--local', '-l', default=None, help='Local env file path (default: searches .env, then ~/.config/cloudmesh/.env)')
@click.option('--remote', '-r', default=None, help='Remote env file path')
def secure_cmd(host: Optional[str], local: Optional[str], remote: Optional[str]):
    """
    Secure env file permissions (600).
    
    Also validates for common security issues.
    If HOST is not specified, local file is secured.
    
    Examples:
        cmc env secure                    # Secure local .env
        cmc env secure uva                # Secure remote .env on uva
        cmc env secure -l .env.production # Secure specific local env
    """
    manager = EnvManager(host=host, local_path=local, remote_path=remote)
    
    target_path = remote if host else local
    success = manager.secure(path=target_path)
    
    if success:
        # Also run validation
        if not host:
            result = manager.validate(target_path)
            if result['warnings']:
                console.print("\n[yellow]Security Warnings:[/yellow]")
                for warning in result['warnings']:
                    console.print(f"  ⚠️  {warning}")
            if result['issues']:
                console.print("\n[red]Issues:[/red]")
                for issue in result['issues']:
                    console.print(f"  ❌ {issue}")


@env_group.command(name="validate")
@click.argument('host', required=False, default=None)
@click.option('--local', '-l', default=None, help='Local env file path (default: searches .env, then ~/.config/cloudmesh/.env)')
@click.option('--remote', '-r', default=None, help='Remote env file path')
@click.option('--fix', is_flag=True, help='Attempt to fix issues (set secure permissions)')
def validate_cmd(host: Optional[str], local: Optional[str], remote: Optional[str], fix: bool):
    """
    Validate .env file for security issues.
    
    Checks permissions, empty secrets, and credential exposure.
    If HOST is not specified, local file is validated.
    
    Examples:
        cmc env validate              # Validate local .env
        cmc env validate uva          # Validate remote .env on uva
        cmc env validate -l .env.production --fix
    """
    manager = EnvManager(host=host, local_path=local, remote_path=remote)
    target_path = remote if host else local
    result = manager.validate(path=target_path)
    
    location = f"{host}:{target_path}" if host else result['path']
    console.print(f"\n[bold]Validation: {location}[/bold]\n")
    
    if result['valid'] and not result['warnings']:
        console.ok("No issues found!")
        return
    
    if result['warnings']:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in result['warnings']:
            console.print(f"  ⚠️  {warning}")
    
    if result['issues']:
        console.print("\n[red]Issues:[/red]")
        for issue in result['issues']:
            console.print(f"  ❌ {issue}")
    
    if fix and result['warnings']:
        # Try to fix permissions
        for warning in result['warnings']:
            if 'permissions' in warning.lower():
                console.print("\n[blue]Fixing permissions...[/blue]")
                manager.secure(path=target_path)
                break


@env_group.command(name="edit")
@click.argument('host', required=False, default=None)
@click.option('--remote', '-r', default=None, help='Remote env file path')
@click.option('--local', '-l', default=None, help='Local env file path to edit (default: searches .env, then ~/.config/cloudmesh/.env)')
def edit_cmd(host: Optional[str], remote: Optional[str], local: Optional[str]):
    """
    Edit .env file using local editor.
    
    If HOST is specified, downloads from remote, opens editor, uploads on save.
    If HOST is not specified, edits local file directly.
    If remote file doesn't exist, offers to create or copy from local.
    
    Requires $EDITOR environment variable or defaults to 'nano'.
    
    Examples:
        cmc env edit                    # Edit local .env file
        cmc env edit uva                # Edit remote .env on uva
        cmc env edit dgx -r ~/.custom.env
        cmc env edit uva -l .env.production  # Copy from local if remote missing
    """
    manager = EnvManager(host=host, local_path=local, remote_path=remote)
    
    if host:
        success = manager.edit_remote(local_path=local)
    else:
        # Edit local file directly
        import subprocess
        target_path = local or manager.local_path
        if not os.path.exists(target_path):
            console.error(f"Local file not found: {target_path}")
            return False
        editor = os.environ.get('EDITOR', 'nano')
        console.print(f"[blue]Opening editor: {editor}[/blue]")
        console.print(f"[dim]File: {target_path}[/dim]\n")
        result = subprocess.run([editor, target_path])
        success = result.returncode == 0
        if success:
            console.ok(f"Edited {target_path}")
        else:
            console.error("Editor exited with error")
    
    if not success:
        console.error("Edit operation failed")


@env_group.command(name="cat")
@click.argument('host', required=False, default=None)
@click.option('--local', '-l', default=None, help='Local env file path (default: searches .env, then ~/.config/cloudmesh/.env)')
@click.option('--remote', '-r', default=None, help='Remote env file path')
def cat_cmd(host: Optional[str], local: Optional[str], remote: Optional[str]):
    """
    Display .env file contents (with security warning).
    
    WARNING: This will display sensitive values including passwords
    and API keys in cleartext. Use with caution.
    
    If HOST is not specified, displays local file.
    
    Examples:
        cmc env cat                    # Display local .env
        cmc env cat uva                # Display remote .env on uva
        cmc env cat dgx -r ~/.custom.env
    """
    # Security warning
    console.warning("⚠️  SECURITY WARNING: This will display all environment variables")
    console.warning("   including passwords, API keys, and secrets in CLEARTEXT.")
    console.print()
    
    if not console.ynchoice("Are you sure you want to continue?", default=False):
        console.print("Cancelled")
        return
    
    manager = EnvManager(host=host, local_path=local, remote_path=remote)
    
    try:
        if host:
            # Read remote file
            content = manager._ssh_download(manager.remote_path)
            console.print(f"\n[bold blue]Remote:[/bold blue] {host}:{manager.remote_path}\n")
        else:
            # Read local file
            target_path = local or manager.local_path
            if not os.path.exists(target_path):
                console.error(f"Local file not found: {target_path}")
                return
            with open(target_path, 'r') as f:
                content = f.read()
            console.print(f"\n[bold blue]Local:[/bold blue] {os.path.abspath(target_path)}\n")
        
        # Display content with syntax highlighting
        console.print("─" * 60)
        console.print(content)
        console.print("─" * 60)
        
    except Exception as e:
        console.error(f"Failed to read file: {e}")


@env_group.command(name="diff")
@click.argument('host')
@click.option('--local', '-l', default=None, help='Local env file path (default: searches .env, then ~/.config/cloudmesh/.env)')
@click.option('--remote', '-r', default=None, help='Remote env file path')
@click.option('--show', is_flag=True, help='Show actual values (not masked)')
def diff_cmd(host: str, local: Optional[str], remote: Optional[str], show: bool):
    """
    Show detailed diff between local and remote .env files.
    
    Similar to probe but focused only on differences.
    
    Examples:
        cmc env diff uva
        cmc env diff dgx --show  # Show actual values (careful with secrets!)
    """
    manager = EnvManager(host=host, local_path=local, remote_path=remote)
    result = manager.probe()
    
    if not result['local_exists']:
        console.error(f"Local file not found: {result['local_path']}")
        return
    
    if not result['remote_exists']:
        console.error(f"Remote file not found on {host}")
        return
    
    diff = result['diff']
    
    if not any([diff['added'], diff['removed'], diff['changed']]):
        console.ok("Files are identical!")
        return
    
    console.print(f"\n[bold]Differences: {local} ↔ {host}:{result['remote_path']}[/bold]\n")
    
    if diff['added']:
        console.print(f"[green]Added in local ({len(diff['added'])}):[/green]")
        for key in diff['added']:
            value = result['local_vars'][key]
            display = value if show else manager._mask_if_secret(key, value)
            console.print(f"  + [green]{key}[/green]={display}")
    
    if diff['removed']:
        console.print(f"\n[red]Removed from local ({len(diff['removed'])}):[/red]")
        for key in diff['removed']:
            value = result['remote_vars'][key]
            display = value if show else manager._mask_if_secret(key, value)
            console.print(f"  - [red]{key}[/red]={display}")
    
    if diff['changed']:
        console.print(f"\n[yellow]Changed ({len(diff['changed'])}):[/yellow]")
        for key in diff['changed']:
            local_val = result['local_vars'][key]
            remote_val = result['remote_vars'][key]
            
            if show:
                local_display = local_val
                remote_display = remote_val
            else:
                local_display = manager._mask_if_secret(key, local_val)
                remote_display = manager._mask_if_secret(key, remote_val)
            
            console.print(f"  ~ [yellow]{key}[/yellow]:")
            console.print(f"      local:  {local_display}")
            console.print(f"      remote: {remote_display}")
    
    console.print("")  # Final newline


def _print_probe_table(result: dict, manager: 'EnvManager'):
    """Pretty print probe results in table format."""
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    
    # Header panel
    status_emoji = {
        'identical': '✅',
        'local_newer': '⬆️',
        'remote_newer': '⬇️',
        'local_only': '📄',
        'missing_local': '❌',
        'missing_remote': '❌',
        'different': '⚠️',
        'remote_error': '💥'
    }.get(result['status'], '?')
    
    status_text = result['status'].replace('_', ' ').title()
    
    title = f"{status_emoji} Environment Comparison: .env ↔ {result['host']}"
    
    # Local file info
    local_info = []
    if result['local_exists']:
        perms = result['local_perms'] or {}
        secure = perms.get('secure', False)
        perm_icon = '✅' if secure else '⚠️'
        local_info.append(f"Path: {result['local_path']}")
        local_info.append(f"Permissions: {perms.get('mode', 'unknown')} {perm_icon}")
        local_info.append(f"Size: {_format_size(result['local_size'])}")
        if result['local_mtime']:
            local_info.append(f"Modified: {result['local_mtime'][:19]}")
        local_info.append(f"Variables: {len(result['local_vars'])}")
    else:
        local_info.append("File not found")
    
    # Remote file info
    remote_info = []
    if result['remote_exists']:
        perms = result['remote_perms'] or {}
        secure = perms.get('secure', False)
        perm_icon = '✅' if secure else '⚠️'
        remote_info.append(f"Path: {result['remote_path']}")
        remote_info.append(f"Permissions: {perms.get('mode', 'unknown')} {perm_icon}")
        remote_info.append(f"Size: {_format_size(result['remote_size'])}")
        if result['remote_mtime']:
            remote_info.append(f"Modified: {result['remote_mtime'][:19]}")
        remote_info.append(f"Variables: {len(result['remote_vars'])}")
    else:
        remote_info.append("File not found")
    
    # Create comparison table
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Variable", style="cyan", no_wrap=True)
    table.add_column("Local", style="blue")
    table.add_column("Remote", style="green")
    table.add_column("Status", style="yellow")
    
    # Combine all keys
    all_keys = set(result['local_vars'].keys()) | set(result['remote_vars'].keys())
    
    for key in sorted(all_keys):
        in_local = key in result['local_vars']
        in_remote = key in result['remote_vars']
        
        if in_local and in_remote:
            local_val = manager._mask_if_secret(key, result['local_vars'][key])
            remote_val = manager._mask_if_secret(key, result['remote_vars'][key])
            
            if result['local_vars'][key] != result['remote_vars'][key]:
                status = "⚠️ CHANGED"
            else:
                status = "✅ SAME"
        elif in_local:
            local_val = manager._mask_if_secret(key, result['local_vars'][key])
            remote_val = "(not set)"
            status = "📝 ADDED"
        else:
            local_val = "(not set)"
            remote_val = manager._mask_if_secret(key, result['remote_vars'][key])
            status = "🗑️ REMOVED"
        
        table.add_row(key, local_val, remote_val, status)
    
    # Summary
    diff = result.get('diff', {})
    summary = f"{len(diff.get('unchanged', []))} unchanged | "
    summary += f"{len(diff.get('changed', []))} changed | "
    summary += f"{len(diff.get('added', []))} added | "
    summary += f"{len(diff.get('removed', []))} removed"
    
    # Build final output
    console.print(f"\n[bold]{title}[/bold]\n")
    
    console.print(Panel(
        "\n".join(local_info),
        title="Local File",
        border_style="blue"
    ))
    
    console.print(Panel(
        "\n".join(remote_info),
        title=f"Remote File ({result['host']})",
        border_style="green"
    ))
    
    if all_keys:
        console.print(table)
    
    console.print(f"\n[dim]{summary}[/dim]")
    console.print(f"Status: {status_text}\n")


def _format_size(size: Optional[int]) -> str:
    """Format byte size to human readable."""
    if size is None:
        return 'unknown'
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def register(cli: Optional[click.Group] = None) -> click.Group:
    """
    Register the env command group.
    
    Args:
        cli: Click group to register with
        
    Returns:
        The env command group
    """
    if cli is not None:
        cli.add_command(env_group, name='env')
    return env_group


# Make available for import
__all__ = ['env_group', 'register', 'EnvManager']