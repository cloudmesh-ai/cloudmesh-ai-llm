import click
from cloudmesh.ai.common.io import console
from cloudmesh.ai.vllm.orchestrator import VLLMOrchestrator

@click.group()
def mock_group():
    """Manage mock vLLM servers."""
    pass

@mock_group.command(name="start")
@click.option("--port", type=int, help="Override both local and remote ports")
@click.argument("name")
def start_mock(name, port):
    """Start a mock vLLM server."""
    try:
        orchestrator = VLLMOrchestrator()
        if orchestrator.prepare_backend(name, port_override=port):
            console.ok(f"Mock backend {name} is ready!")
        else:
            console.error("Mock backend preparation failed.")
    except Exception as e:
        console.error(f"Error orchestrating mock launch: {e}")