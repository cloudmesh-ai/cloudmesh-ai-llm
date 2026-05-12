"""Unit tests for VLLMOrchestrator in cloudmesh.ai.command.launch."""

import pytest
from unittest.mock import MagicMock, patch
from cloudmesh.ai.common.dotdict import DotDict
from cloudmesh.ai.command.launch import VLLMOrchestrator

@pytest.fixture
def mock_orchestrator():
    """Fixture for VLLMOrchestrator with mocked YamlDB."""
    from cloudmesh.ai.vllm.config import VLLMConfig
    VLLMConfig._global_cache = None
    orchestrator = VLLMOrchestrator()
    mock_db = MagicMock()
    orchestrator.db = mock_db
    yield orchestrator, mock_db

@pytest.fixture
def mock_vllm_deps():
    """Fixture to mock vLLM dependencies."""
    with patch("cloudmesh.ai.command.launch.get_default_host") as mock_host, \
         patch("cloudmesh.ai.command.launch.get_server") as mock_server_func, \
         patch("cloudmesh.ai.command.launch.VLLMClient") as mock_client_cls, \
         patch("cloudmesh.ai.common.io.console.ynchoice", return_value=True) as mock_yn:
        
        mock_server = MagicMock()
        mock_server_func.return_value = mock_server
        
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        yield {
            "host": mock_host,
            "server": mock_server,
            "client": mock_client
        }

def test_prepare_backend_already_healthy(mock_orchestrator, mock_vllm_deps):
    """Test that it returns True immediately if server is already alive."""
    orchestrator, mock_db = mock_orchestrator
    
    # Mock DB to return a valid config for 'test-service'
    # VLLMConfig does db.get("cloudmesh") then accesses "cloudmesh.ai.server.{name}"
    # Use a real dictionary for the mock data to avoid MagicMock issues in VLLMConfig
    mock_data = {
        "cloudmesh": {
            "ai": {
                "server": {
                    "test-service": {"host": "test-host", "group": "test-group"}
                }
            }
        },
        "cloudmesh.ai.default.server": "test-service"
    }
    mock_db.get.side_effect = lambda key, default=None: mock_data.get(key, default)
    
    mock_vllm_deps["host"].return_value = "test-host"
    mock_vllm_deps["client"].is_alive.return_value = True
    
    assert orchestrator.prepare_backend("test-service") is True
    
    # Should NOT call start or tunnel
    mock_vllm_deps["server"].start.assert_not_called()
    mock_vllm_deps["server"].tunnel.assert_not_called()

def test_prepare_backend_tunnel_recovery(mock_orchestrator, mock_vllm_deps):
    """Test that it establishes tunnel if server is running but tunnel is down."""
    orchestrator, mock_db = mock_orchestrator
    
    mock_data = {
        "cloudmesh": {
            "ai": {
                "server": {
                    "test-service": {"host": "test-host", "group": "test-group"}
                }
            }
        },
        "cloudmesh.ai.default.server": "test-service"
    }
    mock_db.get.side_effect = lambda key, default=None: mock_data.get(key, default)
    
    mock_vllm_deps["host"].return_value = "test-host"
    
    # First call False (no tunnel), second call True (after tunnel)
    mock_vllm_deps["client"].is_alive.side_effect = [False, True]
    
    assert orchestrator.prepare_backend("test-service") is True
    
    mock_vllm_deps["server"].tunnel.assert_called_once()
    mock_vllm_deps["server"].start.assert_not_called()

def test_prepare_backend_full_start(mock_orchestrator, mock_vllm_deps):
    """Test full pipeline: Tunnel -> Start -> Poll until healthy."""
    orchestrator, mock_db = mock_orchestrator
    
    mock_data = {
        "cloudmesh": {
            "ai": {
                "server": {
                    "test-service": {"host": "test-host", "group": "test-group"}
                }
            }
        },
        "cloudmesh.ai.default.server": "test-service"
    }
    mock_db.get.side_effect = lambda key, default=None: mock_data.get(key, default)
    
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
    orchestrator, mock_db = mock_orchestrator
    
    mock_db.get.side_effect = lambda key, default=None: {
        "cloudmesh": {
            "ai": {
                "server": {
                    "test-service": {"host": "test-host", "group": "test-group"}
                }
            }
        },
        "cloudmesh.ai.default.server": "test-service"
    }.get(key, default)
    
    mock_vllm_deps["host"].return_value = "test-host"
    
    # Always return False
    mock_vllm_deps["client"].is_alive.return_value = False
    
    # We can't wait 2 minutes in a unit test, so we patch time.sleep
    with patch("time.sleep"):
        assert orchestrator.prepare_backend("test-service") is False
    
    mock_vllm_deps["server"].start.assert_called_once()

def test_prepare_backend_no_host(mock_orchestrator, mock_vllm_deps):
    """Test that it raises ValueError if no default host is configured."""
    orchestrator, mock_db = mock_orchestrator
    
    # Mock DB to return a config with NO host
    def mock_get_no_host(key, default=None):
        if key == "cloudmesh":
            return {
                "ai": {
                    "server": {
                        "test-service": {"group": "test-group"}
                    }
                }
            }
        if key == "cloudmesh.ai.default.server":
            return "test-service"
        return default

    mock_db.get.side_effect = mock_get_no_host
    
    mock_vllm_deps["host"].return_value = None
    
    with pytest.raises(ValueError, match="Host not specified"):
        orchestrator.prepare_backend("test-service")
