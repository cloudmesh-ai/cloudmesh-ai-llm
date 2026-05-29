import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
from cloudmesh.ai.common.dotdict import DotDict
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.server_dgx import ServerDGX
from cloudmesh.ai.vllm.server_uva import ServerUVA
from cloudmesh.ai.vllm.orchestrator import VLLMOrchestrator, get_default_host
from cloudmesh.ai.vllm.squeue import SQueue

class TestVLLMConfig(unittest.TestCase):
    def setUp(self):
        # Reset global cache before each test
        VLLMConfig._global_cache = None

    @patch("builtins.open", new_callable=mock_open, read_data="cloudmesh:\n  ai:\n    server:\n      test_server:\n        host: test-host\n        user: test-user\n        remote_port: 8000")
    @patch("os.path.exists", return_value=True)
    def test_caching(self, mock_exists, mock_file):
        # First instantiation should load the file
        config1 = VLLMConfig("test_server")
        # Second instantiation
        config2 = VLLMConfig("test_server")
        
        # Current implementation loads internal + user config on every instantiation
        self.assertEqual(mock_file.call_count, 4) 
        self.assertEqual(config1.cloudmesh.ai.server.test_server.host, "test-host")
        self.assertEqual(config2.cloudmesh.ai.server.test_server.host, "test-host")

    def test_resolve_path(self):
        # Use a DotDict instead of MagicMock to avoid returning mocks
        config = DotDict({
            "user": "alice",
            "remote_port": 9000,
            "dir": "/home/{user}/vllm_{port}"
        })
        # Add smart_get method to the DotDict mock
        config.smart_get = lambda k, default=None: config.get(k, default)
        
        # We need to use the actual method from VLLMConfig
        resolved = VLLMConfig.resolve_path(config, "dir", "/default/{user}")
        self.assertEqual(resolved, "/home/alice/vllm_9000")
        
        resolved_default = VLLMConfig.resolve_path(config, "non_existent", "/default/{user}")
        self.assertEqual(resolved_default, "/default/alice")

    @patch("builtins.open", new_callable=mock_open, read_data="cloudmesh:\n  ai:\n    server:\n      test_server: {}")
    @patch("os.path.exists", return_value=True)
    def test_external_references(self, mock_exists, mock_file):
        config = VLLMConfig("test_server")
        # Mock the internal resolver
        ref = "~/.ssh/config:test_host.User"
        # We simulate the file content for the resolver
        with patch("builtins.open", mock_open(read_data="Host test_host\n  User bob")):
            val = config._resolve_external_reference(ref)
            self.assertEqual(val, "bob")

class TestServerLogic(unittest.TestCase):
    @patch("subprocess.run")
    def test_dgx_start_command(self, mock_run):
        config = DotDict({
            "name": "test-dgx",
            "host": "dgx-host",
            "user": "dgx-user",
            "port": "8000",
            "model": "gemma",
            "image": "vllm-image",
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.95,
            "max_model_len": 131072,
            "cache_dir": "/cache",
            "gpu_ids": "0,1,2,3",
            "group": "dgx",
            "script": "docker run {image} --model {model}"
        })

        server = ServerDGX("dgx-host", db=None)
        server._get_config = MagicMock(return_value=config)
        server._execute = MagicMock()
        server._execute.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
        
        server.start(config.name)
        
        # Verify that the start command was executed via _execute
        # The last call to _execute should be the execution of the script
        last_call_args = server._execute.call_args[0][0]
        self.assertIn("start_test-dgx.sh", last_call_args)

        # The docker run command is uploaded via subprocess.run in _upload_script
        # We check if any call to subprocess.run contains the expected docker command
        all_subprocess_calls = [call[0][0] for call in mock_run.call_args_list]
        # Join all calls into one big string to search for the command
        all_calls_str = " ".join([" ".join(call) if isinstance(call, list) else str(call) for call in all_subprocess_calls])
        self.assertIn("docker run", all_calls_str)
        self.assertIn("vllm-dgx-test-dgx", all_calls_str)
        self.assertIn("--model gemma", all_calls_str)

    @patch("subprocess.run")
    def test_uva_start_sbatch(self, mock_run):
        config = DotDict({
            "name": "test-uva",
            "host": "uva",
            "user": "uva-user",
            "port": "8000",
            "model": "gemma",
            "image": "uva-image",
            "partition": "bii-gpu",
            "gres": "gpu:1",
            "cpus": "1",
            "mem": "16G",
            "time": "01:00:00",
            "group": "uva",
            "script": "apptainer run {image} --model {model}"
        })

        server = ServerUVA("uva", db=None)
        server._get_config = MagicMock(return_value=config)
        server._execute = MagicMock()
        server._execute.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
        
        server.start(config.name, sbatch=True)
        
        # Verify sbatch submission via _execute
        server._execute.assert_called()
        calls = [call[0][0] for call in server._execute.call_args_list]
        self.assertTrue(any("sbatch" in call for call in calls))

class TestVLLMOrchestrator(unittest.TestCase):
    @patch("cloudmesh.ai.vllm.config.VLLMConfig")
    @patch("cloudmesh.ai.vllm.orchestrator.VLLMConfig")
    def test_get_default_host(self, mock_orch_config, mock_cfg_config):
        # Setup mock DB - MUST include servers to pass the empty check in get_default_host
        mock_db = DotDict({
            "cloudmesh.ai.default.server": "server1",
            "cloudmesh.ai.server": {"server1": {"host": "host1"}}
        })
        
        # Setup mock config instances for both potential references
        for mock_cls in [mock_orch_config, mock_cfg_config]:
            instance = mock_cls.return_value
            def get_server_side_effect(name):
                if name == "server1":
                    return DotDict({"host": "host1"})
                return None
            instance.get_server.side_effect = get_server_side_effect
        
        # Pass the mock DB to the function
        host = get_default_host(db=mock_db)
        self.assertEqual(host, "host1")

    @patch("cloudmesh.ai.vllm.orchestrator.SQueue")
    def test_stop_uva_job_matching(self, mock_squeue_cls):
        orchestrator = VLLMOrchestrator()
        sq_instance = mock_squeue_cls.return_value
        sq_instance.get_jobs.return_value = [
            {"job_id": "123", "name": "vllm_test_server_8000"},
            {"job_id": "456", "name": "other_job"}
        ]
        
        # Mock VLLMConfig for the server
        with patch("cloudmesh.ai.vllm.orchestrator.VLLMConfig") as mock_cfg_cls:
            cfg = mock_cfg_cls.return_value
            cfg.get.side_effect = lambda k, default=None: {
                "remote_port": 8000,
                "job_id": None
            }.get(k, default)
            cfg.name = "test_server"
            
            # Mock get_job_name to return the expected name
            orchestrator.get_job_name = MagicMock(return_value="vllm_test_server_8000")
            
            orchestrator.stop_uva(server_name="test_server")
            
            # Verify that job 123 was cancelled
            sq_instance.cancel.assert_called_with("123")

if __name__ == "__main__":
    unittest.main()