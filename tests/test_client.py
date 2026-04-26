"""Unit tests for cloudmesh.ai.vllm.client module."""

import pytest
from unittest.mock import MagicMock, patch
from cloudmesh.ai.vllm.client import VLLMClient

@pytest.fixture
def mock_config():
    """Fixture for a basic vLLM configuration."""
    return {
        "host": "test-host",
        "port": "8000",
        "log_file": "/tmp/vllm.log"
    }

@pytest.fixture
def client(mock_config):
    """Fixture for VLLMClient instance."""
    return VLLMClient(mock_config)

def test_client_init(client, mock_config):
    """Test VLLMClient initialization."""
    assert client.host == mock_config["host"]
    assert client.port == mock_config["port"]
    assert client.api_url == "http://test-host:8000/health"

@patch("requests.get")
def test_is_alive_success(mock_get, client):
    """Test is_alive returns True when both health and models endpoints are 200."""
    # Mock two consecutive calls: 1. /health, 2. /v1/models
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    
    assert client.is_alive() is True
    assert mock_get.call_count == 2

@patch("requests.get")
def test_is_alive_health_fail(mock_get, client):
    """Test is_alive returns False when health endpoint fails."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response
    
    assert client.is_alive() is False
    # Should return False immediately after first call fails
    assert mock_get.call_count == 1

@patch("requests.get")
def test_is_alive_models_fail(mock_get, client):
    """Test is_alive returns False when models endpoint fails."""
    # First call (health) succeeds, second (models) fails
    resp_health = MagicMock(status_code=200)
    resp_models = MagicMock(status_code=404)
    mock_get.side_effect = [resp_health, resp_models]
    
    assert client.is_alive() is False
    assert mock_get.call_count == 2

@patch("requests.get")
def test_is_alive_exception(mock_get, client):
    """Test is_alive returns False on request exception."""
    mock_get.side_effect = Exception("Connection error")
    assert client.is_alive() is False

@patch("cloudmesh.ai.vllm.client.RemoteExecutor")
def test_get_logs_success(mock_remote, client):
    """Test successful log retrieval."""
    mock_executor = MagicMock()
    mock_remote.return_value.__enter__.return_value = mock_executor
    mock_executor.execute.return_value = (0, "log line 1\nlog line 2", "")
    
    logs = client.get_logs(lines=10)
    assert logs == "log line 1\nlog line 2"
    mock_executor.execute.assert_called_with("tail -n 10 /tmp/vllm.log")

@patch("cloudmesh.ai.vllm.client.RemoteExecutor")
def test_get_logs_remote_error(mock_remote, client):
    """Test log retrieval when remote command fails."""
    mock_executor = MagicMock()
    mock_remote.return_value.__enter__.return_value = mock_executor
    mock_executor.execute.return_value = (1, "", "Permission denied")
    
    logs = client.get_logs()
    assert "Error retrieving logs (status 1): Permission denied" in logs

@patch("cloudmesh.ai.vllm.client.RemoteExecutor")
def test_get_logs_exception(mock_remote, client):
    """Test log retrieval on SSH exception."""
    mock_remote.side_effect = Exception("SSH Timeout")
    
    logs = client.get_logs()
    assert "Error retrieving logs: SSH Timeout" in logs

def test_get_logs_no_config(mock_config):
    """Test get_logs when log_file is missing from config."""
    del mock_config["log_file"]
    client = VLLMClient(mock_config)
    assert client.get_logs() == "No log file configured."