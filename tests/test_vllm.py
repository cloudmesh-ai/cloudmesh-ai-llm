import pytest
from unittest.mock import patch, mock_open, MagicMock
from click.testing import CliRunner
from cloudmesh.command.vllm import vllm_group

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

def test_launch_success(runner, mock_creds):
    """Test successful launch command construction."""
    with patch("builtins.open", mock_open(read_data="mock-data")), \
         patch("subprocess.run") as mock_run:
        
        # We need to handle multiple open calls for different files
        # Using a side_effect for open to return different data based on path
        def open_side_effect(path, *args, **kwargs):
            for key, value in mock_creds.items():
                if key in path:
                    return mock_open(read_data=value).return_value
            raise FileNotFoundError(path)
        
        with patch("builtins.open", side_effect=open_side_effect):
            result = runner.invoke(vllm_group, ["launch", "2"])
            
            assert result.exit_code == 0
            assert "LAUNCHING VLLM ON DGX" in result.output
            assert "Container vllm-server started" in result.output
            
            # Verify the docker run command was called
            args, kwargs = mock_run.call_args
            cmd = args[0]
            assert "docker run -d --name vllm-server" in cmd
            assert "--tensor-parallel-size 2" in cmd
            assert "NVIDIA_VISIBLE_DEVICES=0,1" in cmd
            assert "HF_TOKEN=mock-hf-token" in cmd
            assert "VLLM_API_KEY=mock-vllm-key" in cmd

def test_launch_custom_device(runner, mock_creds):
    """Test launch with explicit device IDs."""
    def open_side_effect(path, *args, **kwargs):
        for key, value in mock_creds.items():
            if key in path:
                return mock_open(read_data=value).return_value
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=open_side_effect), \
         patch("subprocess.run") as mock_run:
        
        result = runner.invoke(vllm_group, ["launch", "--device", "4,5,6,7"])
        
        assert result.exit_code == 0
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert "NVIDIA_VISIBLE_DEVICES=4,5,6,7" in cmd
        assert "--tensor-parallel-size 4" in cmd

def test_launch_dryrun(runner, mock_creds):
    """Test dryrun option prints command without executing."""
    def open_side_effect(path, *args, **kwargs):
        for key, value in mock_creds.items():
            if key in path:
                return mock_open(read_data=value).return_value
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=open_side_effect), \
         patch("subprocess.run") as mock_run:
        
        result = runner.invoke(vllm_group, ["launch", "--dryrun"])
        
        assert result.exit_code == 0
        assert "[DRY-RUN]" in result.output
        assert "docker run" in result.output
        mock_run.assert_not_called()

def test_launch_ui(runner, mock_creds):
    """Test launch with UI enabled."""
    def open_side_effect(path, *args, **kwargs):
        for key, value in mock_creds.items():
            if key in path:
                return mock_open(read_data=value).return_value
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=open_side_effect), \
         patch("subprocess.run") as mock_run:
        
        result = runner.invoke(vllm_group, ["launch", "--ui"])
        
        assert result.exit_code == 0
        # Should call subprocess.run twice: once for vllm, once for ui
        assert mock_run.call_count == 2
        
        # Check second call is for UI
        ui_cmd = mock_run.call_args_list[1][0][0]
        assert "docker run -d --name vllm-ui" in ui_cmd
        assert "ghcr.io/open-webui/open-webui:main" in ui_cmd

def test_launch_missing_creds(runner):
    """Test launch failure when credentials are missing."""
    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        result = runner.invoke(vllm_group, ["launch"])
        
        assert result.exit_code == 0 # Click commands often return 0 unless sys.exit is called
        assert "❌ Error: Missing credentials" in result.output

def test_status_running(runner):
    """Test status command when container is running."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="Up 2 hours", returncode=0)
        
        result = runner.invoke(vllm_group, ["status"])
        
        assert result.exit_code == 0
        assert "vLLM is running: Up 2 hours" in result.output

def test_status_not_running(runner):
    """Test status command when container is not running."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        
        result = runner.invoke(vllm_group, ["status"])
        
        assert result.exit_code == 0
        assert "vLLM is not running" in result.output

def test_kill_success(runner):
    """Test kill command success."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        result = runner.invoke(vllm_group, ["kill"])
        
        assert result.exit_code == 0
        assert "Successfully killed vllm-server" in result.output
        args, kwargs = mock_run.call_args
        assert "docker rm -f vllm-server vllm-ui" in args[0]

def test_prompt_success(runner):
    """Test prompt command successful API call."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from vLLM!"}}]
        }
        mock_post.return_value = mock_response
        
        result = runner.invoke(vllm_group, ["prompt", "Hello"])
        
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
        
        result = runner.invoke(vllm_group, ["prompt", "--file", str(prompt_file)])
        
        assert result.exit_code == 0
        assert "File response" in result.output
        
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        assert payload['messages'][0]['content'] == "Prompt from file"

def test_prompt_failure(runner):
    """Test prompt command API failure."""
    with patch("requests.post") as mock_post:
        mock_post.side_effect = Exception("Connection error")
        
        result = runner.invoke(vllm_group, ["prompt", "Hello"])
        
        assert result.exit_code == 0
        assert "Error calling vLLM API: Connection error" in result.output