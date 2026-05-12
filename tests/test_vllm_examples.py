import pytest
import os
from pathlib import Path
from unittest.mock import patch
from cloudmesh.ai.vllm.server_uva import ServerUVA
from cloudmesh.ai.vllm.server_dgx import ServerDGX

def test_server_uva_load_examples(tmp_path):
    # Mock the config path to a temporary directory
    config_file = tmp_path / "vllm_servers.yaml"
    
    with patch("os.path.expanduser", return_value=str(config_file)):
        # Initialize ServerUVA.
        server = ServerUVA("test-host")
        
        # Check if examples are resolvable via VLLMConfig
        # We use _get_config which uses VLLMConfig to merge examples and user config
        config = server._get_config("gemma-4-31b")
        assert config is not None
        assert "model" in config
        assert "image" in config

def test_server_dgx_load_examples(tmp_path):
    # Mock the config path to a temporary directory
    config_file = tmp_path / "vllm_servers.yaml"
    
    with patch("os.path.expanduser", return_value=str(config_file)):
        # Initialize ServerDGX.
        server = ServerDGX("test-host")
        
        # Check if examples are resolvable via VLLMConfig
        config = server._get_config("gemma-4-31b")
        assert config is not None
        assert "model" in config
        assert "image" in config

def test_no_reload_if_config_exists(tmp_path):
    # Create a dummy config file
    config_file = tmp_path / "vllm_servers.yaml"
    config_file.write_text("custom-server: {model: 'custom-model'}")
    
    with patch("os.path.expanduser", return_value=str(config_file)):
        server = ServerUVA("test-host")
        
        # Should have custom-server in the user DB
        assert server.db.get("custom-server") is not None
        # Examples are still resolvable via VLLMConfig, but not present in the user DB
        assert server.db.get("gemma-4-31b") is None
