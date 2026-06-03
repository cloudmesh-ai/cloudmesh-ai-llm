import click
from cloudmesh.ai.command.cline import cline_group
from cloudmesh.ai.command.continue_cmd import continue_group
from cloudmesh.ai.command.mock import mock_group

from cloudmesh.ai.command.vllm.lifecycle import start, stop, kill
from cloudmesh.ai.command.vllm.config import (
    set_default, configure, template, init, reset, get_config, list_config, info_vllm
)
from cloudmesh.ai.command.vllm.utils import processes, status, logs, prompt, install_tool, tunnel_group
from cloudmesh.ai.command.vllm.monitor import monitor_group

@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging to expose raw commands")
@click.pass_context
def llm_group(ctx, debug):
    """LLM management extension."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

# Lifecycle commands
llm_group.add_command(start)
llm_group.add_command(stop)
llm_group.add_command(kill)

# Config commands
llm_group.add_command(set_default, name="default")
llm_group.add_command(configure)
llm_group.add_command(template)
llm_group.add_command(init)
llm_group.add_command(reset)
llm_group.add_command(get_config, name="get")
llm_group.add_command(list_config, name="list")
llm_group.add_command(info_vllm, name="info")

# Utility commands
llm_group.add_command(processes)
llm_group.add_command(status)
llm_group.add_command(logs)
llm_group.add_command(prompt)
llm_group.add_command(install_tool, name="install")
llm_group.add_command(tunnel_group, name="tunnel")

# Groups
llm_group.add_command(monitor_group)
llm_group.add_command(cline_group)
llm_group.add_command(continue_group)

def register(cli=None, **kwargs):
    """Register the llm command group. 
    If cli is None, it's being called as the entry point directly by DelegatingCommand.
    """
    if cli is None:
        # DelegatingCommand passes args and standalone_mode via kwargs
        args = kwargs.get('args')
        standalone_mode = kwargs.get('standalone_mode', True)
        try:
            llm_group.main(args=args, standalone_mode=standalone_mode)
        except Exception as e:
            from cloudmesh.ai.common.io import console
            console.debug(f"Fallback to direct llm_group call due to: {e}")
            llm_group()
        return
    cli.add_command(llm_group, name="llm")
    cli.add_command(mock_group, name="mock")
