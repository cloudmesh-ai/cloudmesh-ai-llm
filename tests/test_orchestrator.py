"""Unit tests for VLLMOrchestrator in cloudmesh.ai.command.launch."""

import pytest
from unittest.mock import MagicMock, patch
from cloudmesh.ai.command.launch import VLLMOrchestrator

@pytest.fixture
def mock_orchestrator():
    """Fixture for VLLMOrchestrator with mocked YamlDB."""
    with patch("cloudmesh.ai.command.launch.YamlDB") as mock_db:
        orchestrator = VLLMOrchestrator()
        yield orchestrator, mock_db.return_value

@pytest.fixture
def mock_vllm_deps():
    """Fixture to mock vLLM dependencies."""
    with patch("cloudmesh.ai.command.launch.get_default_host") as mock_host, \
         patch("cloudmesh.ai.command.launch.get_server") as mock_server_func, \
         patch("cloudmesh.ai.command.launch.VLLMConfig") as mock_config, \
         patch("cloudmesh.ai.command.launch.VLLMClient") as mock_client_cls:
        
        mock_server = MagicMock()
        mock_server_func.return_value = mock_server
        
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        yield {
            "host": mock_host,
            "server": mock_server,
            "client": mock_client,
            "config": mock_config
        }

def test_prepare_backend_already_healthy(mock_orchestrator, mock_vllm_deps):
    """Test that it returns True immediately if server is already alive."""
    orchestrator, _ = mock_orchestrator
    mock_vllm_deps["host"].return_value = "test-host"
    mock_vllm_deps["client"].is_alive.return_value = True
    
    assert orchestrator.prepare_backend("test-service") is True
    
    # Should NOT call start or tunnel
    mock_vllm_deps["server"].start.assert_not_called()
    mock_vllm_deps["server"].tunnel.assert_not_called()

def test_prepare_backend_tunnel_recovery(mock_orchestrator, mock_vllm_deps):
    """Test that it establishes tunnel if server is running but tunnel is down."""
    orchestrator, _ = mock_orchestrator
    mock_vllm_deps["host"].return_value = "test-host"
    
    # First call False (no tunnel), second call True (after tunnel)
    mock_vllm_deps["client"].is_alive.side_effect = [False, True]
    
    assert orchestrator.prepare_backend("test-service") is True
    
    mock_vllm_deps["server"].tunnel.assert_called_once()
    mock_vllm_deps["server"].start.assert_not_called()

def test_prepare_backend_full_start(mock_orchestrator, mock_vllm_deps):
    """Test full pipeline: Tunnel -> Start -> Poll until healthy."""
    orchestrator, _ = mock_orchestrator
    mock_vllm_deps["host"].return_value = "test-host"
    
    # 1. Initial check: False
    # 2. After tunnel: False
    # 3. After start (polling): False, False, True
    mock_vllm_deps["client"].is_alive.side_effect = [False, False, False, False, True]
    
    assert orchestrator.prepare_backend("test-service") is True
    
    mock_vllm_deps["server"].tunnel.assert_called_once()
    mock_vllm_deps["server"].start.assert_called_once()

def test_prepare_backend_timeout(mock_orchestrator, mock_vllm_deps):
    """Test that it returns False if server never becomes healthy."""
    orchestrator, _ = mock_orchestrator
    mock_vllm_deps["host"].return_value = "test-host"
    
    # Always return False
    mock_vllm_deps["client"].is_alive.return_value = False
    
    # We can't wait 2 minutes in a unit test, so we patch time.sleep
    with patch("time.sleep"):
        assert orchestrator.prepare_backend("test-service") is False
    
    mock_vllm_deps["server"].start.assert_called_once()

def test_prepare_backend_no_host(mock_orchestrator, mock_vllm_deps):
    """Test that it raises ValueError if no default host is configured."""
    orchestrator, _ = mock_orchestrator
    mock_vllm_deps["host"].return_value = None
    
    with pytest.raises(ValueError, match="No default host configured"):
        orchestrator.prepare_backend("test-service")