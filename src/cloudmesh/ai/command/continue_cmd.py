import click
import os
from pathlib import Path
from cloudmesh.ai.common.io import console
from cloudmesh.ai.vllm.continue_manager import ContinueManager

@click.group()
def continue_group():
    """Manage Continue AI configuration."""
    pass

@continue_group.command(name="list")
def list_config():
    """List configured models in Continue."""
    manager = ContinueManager()
    config = manager.get_config()
    models = config.get("models", [])
    if not models:
        console.print("No models configured in Continue.")
        return

    console.print(f"{'Name':<30} {'Model':<30} {'Roles'}")
    console.print("-" * 80)
    for m in models:
        name = m.get("name", "N/A")
        model = m.get("model", "N/A")
        roles = ", ".join(m.get("roles", []))
        console.print(f"{name:<30} {model:<30} {roles}")

@continue_group.command(name="probe")
def probe_config():
    """Show values currently set via the GUI."""
    manager = ContinueManager()
    if not manager.verify_installation():
        console.error("Continue extension is not installed or active in VS Code.")
        raise click.ClickException("Continue extension not found. Please install it in VS Code.")
    config = manager.probe()
    console.banner(label="Continue GUI Probe", color="magenta")
    for key, value in config.items():
        console.print(f"{key}: {value}")

@continue_group.command(name="get")
@click.argument("key")
def get_config(key):
    """Retrieve a specific configuration value from Continue config."""
    manager = ContinueManager()
    config = manager.get_config()
    
    # Support nested keys via dot notation (e.g., "name")
    val = config.get(key)
    if val is None:
        console.error(f"Key '{key}' not found in Continue config.")
    else:
        console.print(f"{key}: {val}")

@continue_group.command(name="set")
@click.argument("key")
@click.argument("value")
def set_config(key, value):
    """Update a configuration value in Continue config."""
    manager = ContinueManager()
    config = manager.get_config()
    
    config[key] = value
    manager.set_config(config)
    console.print(f"Set {key} to {value}")

@continue_group.command(name="edit")
def edit_config():
    """Open Continue config.yaml in the default editor."""
    manager = ContinueManager()
    editor = os.environ.get("EDITOR", "vim")
    try:
        os.system(f"{editor} {manager.config_path}")
    except Exception as e:
        console.error(f"Error opening editor: {e}")

@continue_group.command()
@click.argument("profile")
@click.option("--plan", help="Override the planning model")
@click.option("--act", help="Override the acting model")
def sync_profile(profile, plan, act):
    """Sync Continue model settings with a Cloudmesh LLM profile."""
    manager = ContinueManager()
    if not manager.verify_installation():
        console.error("Continue extension is not installed or active in VS Code.")
        raise click.ClickException("Continue extension not found. Please install it in VS Code.")
    try:
        proposed_config, changed = manager.propose_model_update(profile, plan, act)
        
        if not changed:
            console.print(f"No changes needed for profile '{profile}'.")
            return

        # Print proposed changes
        console.print("\nProposed changes to Continue config.yaml:")
        # Simple diff-like output
        old_config = manager.get_config()
        old_models = old_config.get("models", [])
        new_models = proposed_config.get("models", [])

        for i, m in enumerate(new_models):
            if i < len(old_models):
                if m != old_models[i]:
                    console.print(f"Update model {i}: {old_models[i]} -> {m}")
            else:
                console.print(f"Add model {i}: {m}")

        if click.confirm("\nIs this ok?", default=True):
            manager.set_config(proposed_config)
            console.print("Configuration updated successfully.")
        else:
            console.print("Update cancelled.")

    except Exception as e:
        console.error(f"Error syncing profile: {e}")