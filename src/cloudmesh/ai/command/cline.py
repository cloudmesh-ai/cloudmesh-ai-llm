import click
import os
import subprocess
from pathlib import Path
from rich.table import Table
from rich.box import ROUNDED
from cloudmesh.ai.common.io import console
from cloudmesh.ai.common.util import yn_choice
from cloudmesh.ai.vllm.cline_manager import ClineManager

manager = ClineManager()

class ClineGroup(click.Group):
    """Custom group to support 'cmc llm cline <profile>' as a direct call."""
    def resolve_command(self, ctx, args):
        # If the first arg is not a subcommand, we treat it as a profile for update_models
        if args and args[0] not in self.commands:
            # Return (cmd_name, cmd, args) as expected by Click
            cmd_name = "update_models"
            cmd = self.commands.get(cmd_name)
            return cmd_name, cmd, args
        return super().resolve_command(ctx, args)

@click.group(name="cline", cls=ClineGroup)
def cline_group():
    """Manage Cline configuration."""
    pass

@cline_group.command(name="update_models", hidden=True)
@click.argument("profile")
@click.option("--plan", help="Override the plan model")
@click.option("--act", help="Override the act model")
def update_models(profile, plan, act):
    """Set plan and act models based on a profile from llm.yaml."""
    if not manager.verify_installation():
        console.error("Cline extension is not installed or active in VS Code.")
        raise click.ClickException("Cline extension not found. Please install it in VS Code.")
    try:
        current_state, proposed_state, changed = manager.propose_model_update(profile, plan, act)
        
        console.banner(label="Proposed Cline Configuration Changes", color="yellow")
        console.print(f"New Value Source: [bold blue]{manager.config.user_config_path}[/bold blue]")
        console.print()
        
        current_probe = manager.probe(current_state)
        proposed_probe = manager.probe(proposed_state)
        
        table = Table(box=ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="black", width=30)
        table.add_column("Current Value", style="blue")
        table.add_column("New Value", style="green")

        for key, curr_val in current_probe.items():
            new_val = proposed_probe.get(key)
            
            # Add section breaks to match the probe layout
            if key == "Plan Model" or key == "Act Model":
                table.add_section()

            # Mark both values in red if they differ, green if they match
            curr_val_str = str(curr_val)
            new_val_str = str(new_val)
            if curr_val != new_val:
                curr_val_str = f"[{'red'}] {curr_val_str} [/{'red'}]"
                new_val_str = f"[{'red'}] {new_val_str} [/{'red'}]"
            else:
                curr_val_str = f"[{'green'}] {curr_val_str} [/{'green'}]"
                new_val_str = f"[{'green'}] {new_val_str} [/{'green'}]"
            
            table.add_row(key, curr_val_str, new_val_str)

        console.print(table)

        if not changed:
            console.ok("No changes needed. Current configuration matches proposed.")
        elif yn_choice("Would you like to overwrite the current values with the new values?"):
            manager.set_global_state(proposed_state)
            console.ok(f"Successfully updated Cline configuration for profile '{profile}'.")
        else:
            console.msg("Update cancelled.")

    except Exception as e:
        console.error(f"Error updating Cline configuration: {e}")

@cline_group.command(name="probe")
def probe_config():
    """Show values currently set via the GUI."""
    if not manager.verify_installation():
        console.error("Cline extension is not installed or active in VS Code.")
        raise click.ClickException("Cline extension not found. Please install it in VS Code.")
    probe_data = manager.probe()
    console.banner(label="Cline GUI Probe", color="magenta")
    
    table = Table(box=ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="black", width=30)
    table.add_column("Value", style="blue")
    
    table.add_row("Config File", str(manager.global_state_path))
    table.add_section()

    for key, value in probe_data.items():
        if key == "Plan Model":
            table.add_section()
        if key == "Act Model":
            table.add_section()
        table.add_row(key, str(value))
    
    console.print(table)

@cline_group.command(name="get")
@click.argument("key", required=False)
def get_config(key):
    """Retrieve a specific configuration value or all configurations if no key is provided."""
    state = manager.get_global_state()
    if key:
        val = state.get(key)
        if val is not None:
            console.print(f"{key}: {val}")
        else:
            console.error(f"Key '{key}' not found in global configuration.")
    else:
        if not state:
            console.warning("No global configuration found.")
            return
        console.banner(label="Cline Global Configuration")
        for k, v in state.items():
            console.print(f"{k}: {v}")

@cline_group.command(name="set")
@click.argument("key")
@click.argument("value")
def set_config(key, value):
    """Update or create a configuration setting."""
    state = manager.get_global_state()
    state[key] = value
    manager.set_global_state(state)
    console.ok(f"Set {key} = {value}")

@cline_group.command(name="edit")
def edit_config():
    """Open the global configuration file in your default editor."""
    path = manager.global_state_path
    if not path.exists():
        console.error("Configuration file does not exist.")
        return
    
    editor = os.environ.get("EDITOR", "vim")
    try:
        subprocess.run([editor, str(path)], check=True)
        console.ok("Configuration file edited.")
    except Exception as e:
        console.error(f"Could not open editor {editor}: {e}")

@cline_group.group(name="secrets")
def secrets_group():
    """Manage Cline secrets."""
    pass

@secrets_group.command(name="set")
@click.argument("key")
@click.argument("value")
def set_secret(key, value):
    """Securely set a secret."""
    manager.set_secret(key, value)
    console.ok(f"Secret {key} set.")

@secrets_group.command(name="get")
@click.argument("key", required=False)
def get_secret(key):
    """Retrieve a specific secret or all secrets if no key is provided."""
    if key:
        val = manager.get_secret(key)
        if val:
            console.print(f"{key}: {val}")
        else:
            console.error(f"Secret '{key}' not found.")
    else:
        secrets = manager.get_all_secrets()
        if not secrets:
            console.warning("No secrets stored.")
            return
        console.banner(label="Cline Secrets")
        for k, v in secrets.items():
            console.print(f"{k}: {v}")

cline_group.add_command(secrets_group)

