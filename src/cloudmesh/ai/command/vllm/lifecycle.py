import click
import yaml
from pathlib import Path
from cloudmesh.ai.common.io import console
from cloudmesh.ai.common import banner
from cloudmesh.ai.vllm.orchestrator import VLLMOrchestrator, get_server, get_vllm_api_key
from cloudmesh.ai.vllm.tunnel import tunnel_manager
from cloudmesh.ai.vllm.aider_launcher import AiderLauncher
from cloudmesh.ai.vllm.webui_launcher import WebUILauncher
from cloudmesh.ai.vllm.claude_launcher import ClaudeLauncher
from cloudmesh.ai.vllm.exceptions import VLLMError

@click.command(name="start")
@click.option("--ui", is_flag=True, help="Launch WebUI after backend is ready")
@click.option("--claude", is_flag=True, help="Launch Claude after backend is ready")
@click.option("--info", is_flag=True, help="Display server configuration info")
@click.option("--export", is_flag=True, help="Export launch scripts to local directory for customization")
@click.option("--port", type=int, help="Override both local and remote ports")
@click.option("--device", help="Explicit device IDs (e.g. '0,1,2,3')")
@click.option("--dryrun", is_flag=True, help="Print command without executing")
@click.argument("name")
@click.pass_context
def start(ctx, name, ui, claude, info, export, port, device, dryrun):
    """Full pipeline: Start vLLM server -> Tunnel -> Health Check -> Optional UI."""
    try:
        debug = ctx.obj.get("debug", False)
        if info:
            orchestrator = VLLMOrchestrator(debug=debug)
            servers = orchestrator.config.get("cloudmesh.ai.server", {})
            config_path = orchestrator.config_path
            if isinstance(servers, dict) and name in servers:
                config = servers[name]
                info_text = f"Config File: {config_path}\n"
                info_text += "\n".join([f"{k}: {v}" for k, v in config.items()])
                console.print(banner(f"Server Info: {name}", info_text))
            else:
                available = list(servers.keys()) if isinstance(servers, dict) else []
                error_msg = (
                    f"Server '{name}' not found in configuration.\n"
                    f"Available servers: {', '.join(available) if available else 'None'}"
                )
                console.error(error_msg)
                
                try:
                    console.print(banner("Configuration File Location", config_path))
                    with open(config_path, 'r') as f:
                        content = f.read()
                        console.print(banner("Configuration File Contents", content))
                except Exception as e:
                    console.error(f"Could not read configuration file: {e}")
            return

        orchestrator = VLLMOrchestrator(debug=debug)

        if export:
            console.print(f"[blue]Exporting scripts for {name}...[/blue]")
            if orchestrator.export_scripts(name):
                console.ok("Scripts exported successfully. You can now edit them locally.")
            else:
                console.error("Failed to export scripts.")
            return
        
        # Check if the name refers to a client instead of a server
        clients = orchestrator.config.get("cloudmesh.ai.client", {})
        # Recognize common clients even if not explicitly in config
        if (isinstance(clients, dict) and name in clients) or name in ["aider", "webui", "claude", "openwebui"]:
            # Normalize name for launcher lookup
            norm_name = "webui" if name == "openwebui" else name
            client_config = clients.get(name, {}) if isinstance(clients, dict) else {}
            
            # 1. Resolve Port (incorporating logic from 'launch' command)
            if port:
                client_config["port"] = port
            elif not client_config.get("port"):
                # Fallback: find a running server port
                default_server = orchestrator.config.get("cloudmesh.ai.default.server")
                servers = orchestrator.config.get("cloudmesh.ai.server", {})
                local_port = 8000
                if default_server and isinstance(servers, dict):
                    local_port = servers.get(default_server, {}).get("local_port", 8000)
                elif isinstance(servers, dict):
                    for s_name, s_cfg in servers.items():
                        if isinstance(s_cfg, dict) and s_cfg.get("job_id"):
                            local_port = s_cfg.get("local_port", 8000)
                            break
                client_config["port"] = local_port

            # 2. Resolve API Key
            raw_key = client_config.get("OPENAI_API_KEY") or client_config.get("openai_api_key")
            if not raw_key:
                api_key = get_vllm_api_key(orchestrator.config)
                if api_key:
                    client_config["OPENAI_API_KEY"] = api_key
            elif raw_key.startswith("{") and raw_key.endswith("}"):
                lookup_key = raw_key[1:-1]
                api_key = get_vllm_api_key(orchestrator.config, lookup_key=lookup_key)
                if api_key:
                    client_config["OPENAI_API_KEY"] = api_key
            
            # 3. Resolve Launcher and Config
            launcher_name = client_config.get("launcher") or norm_name
            
            # Special handling for Aider template if not fully configured
            if norm_name == "aider" and not client_config.get("model"):
                try:
                    config_path = Path(__file__).parent / ".." / "vllm" / "config" / "templates" / "aider.yaml"
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            tpl = yaml.safe_load(f)
                        aider_tpl = tpl.get("cloudmesh", {}).get("ai", {}).get("aider", {})
                        for k, v in aider_tpl.items():
                            client_config.setdefault(k, v)
                except (yaml.YAMLError, OSError) as e:
                    console.debug(f"Could not load Aider template from {config_path}: {e}")

            console.print(banner(f"Launching Client: {name}", 
                                  f"Host: {client_config.get('host', 'localhost')}\n"
                                  f"Port: {client_config.get('port', '8000')}\n"
                                  f"Launcher: {launcher_name}"))
            
            launchers = {
                "webui": WebUILauncher,
                "openwebui": WebUILauncher,
                "claude": ClaudeLauncher,
                "aider": AiderLauncher,
            }
            
            launcher_class = launchers.get(launcher_name)
            if launcher_class:
                launcher_class().launch(client_config=client_config)
            else:
                console.error(f"Unsupported or missing launcher '{launcher_name}' for client '{name}'.")
            return

        if orchestrator.prepare_backend(name, port_override=port):
            console.ok(f"Backend {name} is ready!")
            
            if ui:
                console.print("[bold green]Launching WebUI...[/bold green]")
                clients = orchestrator.config.get("cloudmesh.ai.client", {})
                webui_cfg = clients.get("openwebui", {}) if isinstance(clients, dict) else {}
                WebUILauncher().launch(client_config=webui_cfg)
            elif claude:
                console.print("[bold green]Launching Claude...[/bold green]")
                clients = orchestrator.config.get("cloudmesh.ai.client", {})
                claude_cfg = clients.get("claude", {}) if isinstance(clients, dict) else {}
                ClaudeLauncher().launch(client_config=claude_cfg)
            else:
                servers = orchestrator.config.get("cloudmesh.ai.server", {})
                config = servers.get(name, {}) if isinstance(servers, dict) else {}
                model_name = config.get("model", "Unknown Model")
                actual_port = port or config.get("remote_port", 8000)
                
                banner_text = f"Model: {model_name}\nPort: {actual_port}"
                console.banner(label="llm service started", txt=banner_text, color="dark_green")
                
                console.msg("Backend is ready. You can now run 'cmc launch webui' or 'cmc launch claude'.")
                console.print(f"\n[dim]To stop this server, run: cmc llm stop {name} --port {actual_port}[/dim]")
        else:
            console.error("Backend preparation failed.")
    except Exception as e:
        import traceback
        console.error(f"Error orchestrating vLLM launch: {e}")
        console.print(traceback.format_exc())

@click.command(name="stop")
@click.argument("identifier", required=False)
@click.option("--port", type=str, help="Port or partial port (e.g. '123') to identify the job")
@click.pass_context
def stop(ctx, identifier, port):
    """Stop a vLLM server (UVA HPC specific). Supports JobID, fuzzy port, or config port."""
    try:
        debug = ctx.obj.get("debug", False)
        orchestrator = VLLMOrchestrator(debug=debug)
        
        if identifier:
            if identifier.isdigit():
                if orchestrator.stop_uva(port_pattern=identifier):
                    console.ok(f"Successfully stopped server matching {identifier}.")
                    WebUILauncher().stop()
                    return
            if orchestrator.stop_uva(server_name=identifier, port_pattern=port):
                console.ok(f"Successfully stopped server {identifier}.")
                WebUILauncher().stop()
                return
        elif port:
            if orchestrator.stop_uva(port_pattern=port):
                console.ok(f"Successfully stopped server matching port {port}.")
                WebUILauncher().stop()
                return
        else:
            servers = orchestrator.config.get("cloudmesh.ai.server", {})
            if not servers:
                console.error("No servers configured. Use 'cmc llm start <name>' first.")
                return
            last_server = None
            for name, cfg in servers.items():
                if cfg.get("job_id"):
                    last_server = name
                    break
            if last_server:
                if orchestrator.stop_uva(server_name=last_server):
                    console.ok(f"Successfully stopped last started server: {last_server}.")
                WebUILauncher().stop()
                return
            console.error("No active server found in configuration to stop.")
    except Exception as e:
        console.error(f"Error stopping vLLM server: {e}")

@click.command(name="kill")
@click.argument("name")
@click.option("--tunnel", is_flag=True, help="Close the SSH tunnel after killing")
@click.pass_context
def kill(ctx, name, tunnel):
    """Forcefully kill vLLM server."""
    try:
        debug = ctx.obj.get("debug", False)
        orchestrator = VLLMOrchestrator(debug=debug)
        identity = orchestrator.config.resolve_server_identity(name)
        target_host = identity["host"]
        
        server = get_server(target_host)
        server.kill(name)
        
        if tunnel:
            port = identity["port"]
            success, result = tunnel_manager.stop_tunnel(target_host, port)
            if success:
                console.ok(f"Tunnel closed (PID: {result})")
            else:
                console.warning(f"Could not close tunnel: {result}")
            
        console.ok(f"Successfully killed vLLM server '{name}' on {target_host}")
    except Exception as e:
        console.error(f"Error killing vLLM server: {e}")