import pytest
from unittest.mock import patch, mock_open, MagicMock
from click.testing import CliRunner
from cloudmesh.ai.command.vllm import llm_group

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def mock_creds():
    """Mocks the credential files."""
    return {
        "gemma/hugging.txt": "mock-hf-token",
        "gemma/server_master_key.txt": "mock-vllm-key"
    }

@pytest.fixture
def mock_db():
    """Mocks the configuration database."""
    from cloudmesh.ai.common import DotDict
    
    # Use a real DotDict for the mock data
    servers = {
        "2": {"host": "dgx-host", "model": "gemma-2b", "image": "vllm-image"},
        "test-server": {"host": "test-host", "model": "gemma-2b", "image": "vllm-image"},
    }
    
    mock_data = DotDict({
        "cloudmesh": {
            "ai": {
                "server": servers
            }
        },
        "cloudmesh.ai.default.server": "2",
        "config.default_service": "2",
    })

    # Patch VLLMConfig to use this mock_data as its db
    with patch("cloudmesh.ai.vllm.config.VLLMConfig.__init__", 
               lambda self, name, db=None: setattr(self, 'db', mock_data)):
        # We also need to mock the .get() method of VLLMConfig if it's used as a dict
        # but VLLMConfig usually inherits from DotDict or implements __getitem__
        yield mock_data

def test_launch_success(runner, mock_creds, mock_db):
    """Test successful launch command construction."""
    with patch("cloudmesh.ai.command.vllm.RemoteExecutor") as mock_executor, \
         patch("cloudmesh.ai.command.vllm.IJob") as mock_ijob:
        
        # Mock IJob.get().username()
        mock_ijob_instance = mock_ijob.return_value.get.return_value
        mock_ijob_instance.username.return_value = "testuser"
        
        def open_side_effect(path, *args, **kwargs):
            path_str = str(path)
            for key, value in mock_creds.items():
                if key in path_str:
                    return mock_open(read_data=value).return_value
            raise FileNotFoundError(path)
        
        with patch("builtins.open", side_effect=open_side_effect):
            result = runner.invoke(llm_group, ["start", "2"], input="\n")
            
            assert result.exit_code == 0
            assert "Successfully started vLLM server" in result.output
            
            # Verify RemoteExecutor was used
            mock_executor.assert_called()

def test_launch_custom_device(runner, mock_creds, mock_db):
    """Test launch with explicit device IDs."""
    def open_side_effect(path, *args, **kwargs):
        path_str = str(path)
        for key, value in mock_creds.items():
            if key in path_str:
                return mock_open(read_data=value).return_value
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=open_side_effect), \
         patch("cloudmesh.ai.command.vllm.RemoteExecutor") as mock_executor, \
         patch("cloudmesh.ai.command.vllm.IJob") as mock_ijob:
        
        mock_ijob.return_value.get.return_value.username.return_value = "testuser"
        
        result = runner.invoke(llm_group, ["start", "test-server", "--device", "4,5,6,7"], input="\n")
        
        assert result.exit_code == 0
        mock_executor.assert_called()

def test_launch_dryrun(runner, mock_creds, mock_db):
    """Test dryrun option prints command without executing."""
    def open_side_effect(path, *args, **kwargs):
        path_str = str(path)
        for key, value in mock_creds.items():
            if key in path_str:
                return mock_open(read_data=value).return_value
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=open_side_effect), \
         patch("cloudmesh.ai.command.vllm.RemoteExecutor") as mock_executor, \
         patch("cloudmesh.ai.command.vllm.IJob") as mock_ijob:
        
        mock_ijob.return_value.get.return_value.username.return_value = "testuser"
        
        result = runner.invoke(llm_group, ["start", "test-server", "--dryrun"])
        
        assert result.exit_code == 0
        # Verify RemoteExecutor was NOT called
        mock_executor.assert_not_called()

def test_launch_ui(runner, mock_creds, mock_db):
    """Test launch with UI enabled."""
    def open_side_effect(path, *args, **kwargs):
        path_str = str(path)
        for key, value in mock_creds.items():
            if key in path_str:
                return mock_open(read_data=value).return_value
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=open_side_effect), \
         patch("cloudmesh.ai.command.vllm.RemoteExecutor") as mock_executor, \
         patch("cloudmesh.ai.command.vllm.select_vllm_service") as mock_select, \
         patch("cloudmesh.ai.command.vllm.IJob") as mock_ijob:
        
        mock_ijob.return_value.get.return_value.username.return_value = "testuser"
        mock_select.return_value = ("test-server", None, "test-host")
        result = runner.invoke(llm_group, ["start", "--ui"], input="\n")
        
        assert result.exit_code == 0
        mock_executor.assert_called()

def test_launch_missing_creds(runner, mock_db):
    """Test launch failure when credentials are missing."""
    with patch("cloudmesh.ai.command.vllm.RemoteExecutor") as mock_executor:
        # Simulate a remote failure (e.g., cat failing because file is missing)
        mock_executor.return_value.__enter__.return_value.execute.side_effect = Exception("File not found")
        
        result = runner.invoke(llm_group, ["start", "test-server"], input="\n")
        
        assert result.exit_code == 0 # Click commands often return 0 unless sys.exit is called
        assert "Error starting vLLM server: File not found" in result.output

def test_status_running(runner, mock_db):
    """Test status command when container is running."""
    with patch("cloudmesh.ai.vllm.client.VLLMClient.get_status") as mock_status:
        mock_status.return_value = "Up 2 hours"
        
        result = runner.invoke(llm_group, ["status", "test-server"])
        
        assert result.exit_code == 0
        assert "Up 2 hours" in result.output

def test_status_not_running(runner, mock_db):
    """Test status command when container is not running."""
    with patch("cloudmesh.ai.vllm.client.VLLMClient.get_status") as mock_status:
        mock_status.return_value = "Stopped"
        
        result = runner.invoke(llm_group, ["status", "test-server"])
        
        assert result.exit_code == 0
        assert "Stopped" in result.output

def test_kill_success(runner, mock_db):
    """Test kill command success."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        result = runner.invoke(llm_group, ["kill", "test-server"])
        
        assert result.exit_code == 0
        assert "Successfully killed vLLM server" in result.output
        # The actual command uses the container name from ServerDGX
        # which is vllm-dgx-test-server
        args, kwargs = mock_run.call_args
        assert "docker rm -f vllm-dgx-test-server" in args[0]

def test_prompt_success(runner):
    """Test prompt command successful API call."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from vLLM!"}}]
        }
        mock_post.return_value = mock_response
        
        result = runner.invoke(llm_group, ["prompt", "Hello"])
        
        assert result.exit_code == 0
        assert "vLLM Response:" in result.output
        assert "Hello from vLLM!" in result.output
        
        # Verify payload
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        assert payload['messages'][0]['content'] == "Hello"

def test_prompt_file(runner, tmp_path):
    """Test prompt command using a file."""
    # Create a real temporary file so click.Path(exists=True) passes
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_file.write_text("Prompt from file")
    
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "File response"}}]
        }
        mock_post.return_value = mock_response
        
        result = runner.invoke(llm_group, ["prompt", "--file", str(prompt_file)])
        
        assert result.exit_code == 0
        assert "File response" in result.output
        
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        assert payload['messages'][0]['content'] == "Prompt from file"

def test_prompt_failure(runner):
    """Test prompt command API failure."""
    with patch("requests.post") as mock_post:
        mock_post.side_effect = Exception("Connection error")
        
        result = runner.invoke(llm_group, ["prompt", "Hello"])
        
        assert result.exit_code == 0
        assert "Error calling vLLM API: Connection error" in result.output