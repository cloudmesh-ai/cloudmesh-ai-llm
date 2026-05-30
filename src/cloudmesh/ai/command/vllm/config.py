import click
import os
import yaml
from pathlib import Path
from cloudmesh.ai.common import DotDict, banner
from cloudmesh.ai.common.io import console
from cloudmesh.ai.vllm.orchestrator import VLLMOrchestrator
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.templates_manager import get_templates_manager

@click.command(name="get")
@click.argument("name")
@click.pass_context
def get_config(ctx, name):
    """Get configuration for a specific LLM profile (server or client)."""
    try:
        debug = ctx.obj.get("debug", False)
        orchestrator = VLLMOrchestrator(debug=debug)
        config_path = orchestrator.config.config_path
        
        # Try to get as server first, then as client
        profile_data = orchestrator.config.get_server(name)
        profile_type = "server" if profile_data else None
        
        if not profile_data:
            profile_data = orchestrator.config.get_client(name)
            profile_type = "client" if profile_data else None

        if profile_data:
            banner_title = f"LLM {profile_type.capitalize()} Profile: {name}"
            data_to_print = profile_data.to_dict() if hasattr(profile_data, 'to_dict') else profile_data
            
            lines = [f"Config File: {config_path}"]
            lines.extend([f"{k}: {v}" for k, v in data_to_print.items()])
            formatted_data = "\n".join(lines)
            
            console.print(banner(banner_title, formatted_data))
        else:
            # Collect available profiles for a helpful error message
            servers = orchestrator.config.get("cloudmesh.ai.server", {})
            clients = orchestrator.config.get("cloudmesh.ai.client", {})
            available = []
            if isinstance(servers, dict): available.extend([f"server.{k}" for k in servers.keys()])
            if isinstance(clients, dict): available.extend([f"client.{k}" for k in clients.keys()])
            
            error_msg = f"Profile '{name}' not found in configuration."
            if available:
                error_msg += "\nAvailable profiles:\n" + "\n".join(sorted(available))
            console.error(error_msg)

    except Exception as e:
        console.error(f"Error retrieving profile '{name}': {e}")

@click.command(name="list")
@click.argument("key", required=False)
@click.pass_context
def list_config(ctx, key):
    """List configuration leaf names or display merged config for a specific item."""
    try:
        debug = ctx.obj.get("debug", False)
        orchestrator = VLLMOrchestrator(debug=debug)
        db = orchestrator.config.yaml_data
        
        def get_leaf_names(data, prefix=""):
            leaves = []
            if isinstance(data, dict):
                for k, v in data.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict) and v:
                        leaves.extend(get_leaf_names(v, new_prefix))
                    else:
                        leaves.append(new_prefix)
            return leaves

        if not key:
            ai_config = db.get("cloudmesh", {}).get("ai", {})
            if not ai_config:
                return
            
            leaves = sorted(get_leaf_names(ai_config))
            for leaf in leaves:
                console.print(leaf)
            return

        # Resolve the path in the config (using DotDict's smart get)
        data = orchestrator.config.get(f"cloudmesh.ai.{key}")
        if data is None:
            data = orchestrator.config.get(f"cloudmesh.ai.server.{key}")
        
        if data is None:
            return

        if isinstance(data, dict):
            if key in ["server", "client"] or (not "." in key and not any(k in key for k in ["host", "port", "model"])):
                children = sorted(data.keys())
                for child in children:
                    console.print(child)
            else:
                # Show merged data for a specific item
                item_name = key.replace("server.", "").replace("client.", "")
                config_obj = VLLMConfig(item_name)
                console.print(config_obj.yaml_data)
        else:
            console.print(data)

    except Exception as e:
        console.debug(f"Error listing config for key {key}: {e}")

@click.command(name="info")
def info_vllm():
    """List all running vLLM servers (alias for 'status')."""
    # Redirect to status without arguments
    ctx = click.get_current_context()
    # Note: status is defined in utils.py, will be handled by the group registration
    ctx.invoke(status, name=None)

@click.command(name="default")
@click.argument("type", type=click.Choice(['server', 'client'], case_sensitive=False))
@click.argument("name")
@click.pass_context
def set_default(ctx, type, name):
    """Set the default server or client. Usage: cmc llm default [server|client] [NAME]"""
    try:
        debug = ctx.obj.get("debug", False)
        orchestrator = VLLMOrchestrator(debug=debug)
        db = orchestrator.config.yaml_data
        
        # Ensure nested structure exists
        if "cloudmesh" not in db: db["cloudmesh"] = {}
        if "ai" not in db["cloudmesh"]: db["cloudmesh"]["ai"] = {}
        if "default" not in db["cloudmesh"]["ai"]: db["cloudmesh"]["ai"]["default"] = {}
        db["cloudmesh"]["ai"]["default"][type] = name
        
        with open(orchestrator.config.config_path, 'w') as f:
            yaml.dump(db, f)
            
        console.ok(f"Default {type} set to: {name}")
    except Exception as e:
        console.error(f"Error setting default {type}: {e}")

@click.command(name="configure")
@click.pass_context
def configure(ctx):
    """Interactively configure vLLM settings."""
    try:
        debug = ctx.obj.get("debug", False)
        orchestrator = VLLMOrchestrator(debug=debug)
        db = orchestrator.config.yaml_data
        config_path = orchestrator.config.config_path
        
        current_server = orchestrator.config.get("cloudmesh.ai.default.server")
        
        console.banner("vLLM Configuration", f"Config File: {config_path}\nCurrent Default Server: [bold]{current_server or 'Not set'}[/bold]")
        
        servers = orchestrator.config.get("cloudmesh.ai.server", {})
        if servers:
            console.print("\n[bold]Available servers in llm.yaml:[/bold]")
            for name in sorted(servers.keys()):
                marker = "[green]✓[/green]" if name == current_server else " "
                console.print(f" {marker} {name}")
        else:
            console.warning("No servers found in llm.yaml.")

        console.print("\n")
        prompt_text = f"Enter default server [{current_server}]: " if current_server else "Enter default server: "
        new_server = input(prompt_text).strip()
        
        if new_server and new_server != current_server:
            if "cloudmesh" not in db: db["cloudmesh"] = {}
            if "ai" not in db["cloudmesh"]: db["cloudmesh"]["ai"] = {}
            if "default" not in db["cloudmesh"]["ai"]: db["cloudmesh"]["ai"]["default"] = {}
            db["cloudmesh"]["ai"]["default"]["server"] = new_server
            
            with open(config_path, 'w') as f:
                yaml.dump(db, f)
                
            console.ok(f"Default server updated to: {new_server}")
        elif not new_server and not current_server:
            console.error("A server must be specified.")
        else:
            console.msg("No changes made to configuration.")
            
    except Exception as e:
        console.error(f"Error during configuration: {e}")

@click.command(name="template")
@click.argument("template_name", required=False)
def template(template_name):
    """List or apply a vLLM configuration template (e.g., gemma, llama)."""
    manager = get_templates_manager()
    if not template_name:
        templates = manager.list_templates()
        if not templates:
            console.msg("No templates available.")
            return
        console.print(banner("Available vLLM Templates", "\n".join(templates)))
    else:
        if manager.apply_template(template_name):
            console.ok(f"Template '{template_name}' applied successfully.")
        else:
            console.error(f"Failed to apply template '{template_name}'.")

@click.command(name="init")
def init():
    """Initialize vLLM server configurations with defaults for DGX and UVA."""
    config_path = VLLMConfig.DEFAULT_USER_CONFIG_PATH
    
    if os.path.exists(config_path):
        console.warning(f"Configuration file already exists at {config_path}")
        if not console.ynchoice("Do you want to overwrite it with default configurations?", default=False):
            console.msg("Initialization cancelled.")
            return

    if VLLMConfig.reset():
        console.ok("vLLM server configurations initialized successfully!")
        console.banner("Welcome to vLLM Management", 
            "Default configurations for DGX and UVA have been added to your config.\n"
            "You can now use 'cmc llm start [NAME]' to launch a server.")
    else:
        console.error("Error: Could not find default configuration file to initialize from.")

@click.command(name="reset")
def reset():
    """Reset the vLLM server configuration YAML file to the default version."""
    if console.ynchoice("This will overwrite your current vLLM server configurations. Are you sure you want to proceed?", default=False):
        if VLLMConfig.reset():
            console.ok("vLLM server configuration has been reset to defaults.")
        else:
            console.error("Error: Could not find default configuration file to reset from.")
    else:
        console.warning("Reset cancelled.")