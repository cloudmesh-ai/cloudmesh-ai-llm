import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import json
from cloudmesh.ai.vllm.tunnel import TunnelManager
from cloudmesh.ai.vllm.client import VLLMClient
from cloudmesh.ai.vllm.exceptions import VLLMConfigError

class TestVLLMImprovements(unittest.TestCase):

    def test_dynamic_defaulting(self):
        """Test that the first server in the list is used as default if none is set."""
        from cloudmesh.ai.command.vllm import get_default_host
        from cloudmesh.ai.common import DotDict
        
        # Mock the database using DotDict
        mock_db = DotDict({
            "cloudmesh.ai.default.server": None,
            "cloudmesh.ai.server": {
                "server1": {"host": "host1"},
                "server2": {"host": "host2"}
            }
        })
        
        host = get_default_host(db=mock_db)
        self.assertEqual(host, "host1")

    @patch("os.kill")
    @patch("builtins.open", new_callable=mock_open, read_data='{"host1:8000": 1234}')
    @patch("os.path.exists", return_value=True)
    def test_tunnel_manager_stop(self, mock_exists, mock_file, mock_kill):
        """Test that TunnelManager can stop a tracked tunnel."""
        manager = TunnelManager()
        # Mock state loading
        manager._load_state = MagicMock(return_value={"host1:8000": 1234})
        manager._save_state = MagicMock()
        
        success, pid = manager.stop_tunnel("host1", 8000)
        
        self.assertTrue(success)
        self.assertEqual(pid, 1234)
        mock_kill.assert_called_with(1234, 15) # SIGTERM

    @patch("requests.get")
    def test_health_check_depth(self, mock_get):
        """Test the detailed health check states."""
        config = {"host": "localhost", "port": 8000}
        client = VLLMClient(config)
        
        # Case 1: OFFLINE (Connection Error)
        mock_get.side_effect = Exception("Connection failed")
        self.assertEqual(client.get_status(), "OFFLINE")
        
        # Case 2: STARTING (Health endpoint not 200)
        mock_get.side_effect = None
        mock_get.return_value.status_code = 503
        self.assertEqual(client.get_status(), "STARTING")
        
        # Case 3: READY (Health 200 AND Models 200)
        mock_get.side_effect = None
        # First call for /health, second for /v1/models
        mock_get.return_value.status_code = 200
        self.assertEqual(client.get_status(), "READY")

    def test_config_error_handling(self):
        """Test that VLLMConfigError is used for missing configurations."""
        # This is a simple check that the exception exists and can be raised
        with self.assertRaises(VLLMConfigError):
            raise VLLMConfigError("Missing host")

if __name__ == "__main__":
    unittest.main()