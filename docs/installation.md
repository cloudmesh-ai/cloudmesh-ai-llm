# Installation

This guide will walk you through installing the Cloudmesh AI LLM package and configuring your environment for remote GPU access.

## System Prerequisites

!!! attention
    The current version requires cloudmesh-ai-vpn. We may change this in the future 
    if there is demand for not using it.

    Many institutions provide an institutional VPN which you can probably use. If you prefer using your own VPN connection tools, let us know and we will integrate an option for it.
    However, in many cases, you can probably use our VPN connection strategy.
    
Before installing the Python packages, ensure you have the necessary system tools installed for VPN and tunnel connectivity. Cloudmesh VPN documentation is available at <https://cloudmesh-ai.github.io/cloudmesh-ai-vpn/> if you need more details, such as creating a setup for your organization. 

### OpenConnect 
The `cloudmesh-ai-vpn` package requires the `openconnect` binary to establish VPN connections.

**macOS (via Homebrew):**
```bash
brew install openconnect
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install openconnect
```

**Windows (via Chocolatey):**
```powershell
choco install openconnect
```

## Installing the Package

### Via PyPI

!!! attention
    Currently, the package is not yet deployed to PyPI, but once available you can install it using:

```bash
pip install cloudmesh-ai-llm
```

### Via GitHub
You can install the latest versions directly from the GitHub repositories:

```bash
pip install git+https://github.com/cloudmesh-ai/cloudmesh-ai-vpn.git
pip install git+https://github.com/cloudmesh-ai/cloudmesh-ai-llm.git
```

### Source code install

```bash
git clone https://github.com/cloudmesh-ai/cloudmesh-ai-vpn.git
git clone https://github.com/cloudmesh-ai/cloudmesh-ai-llm.git

cd cloudmesh-ai-vpn
pip install -e .
cd ..

cd cloudmesh-ai-llm
pip install -e .
cd ..
```

## SSH Configuration

This configuration is essential for the `cmc llm start` command to successfully orchestrate remote deployments.

To simplify connecting to your GPU nodes, we use the 
`.ssh/config` file. This allows you to use short names like `ssh uva` instead of typing the full IP address and username every time.
Add the following to your `~/.ssh/config` (replacing the placeholders with your actual server details):

```text
Host uva
    HostName <uva-ip-or-hostname>
    User <your-user>

Host spark
    HostName <spark-ip-or-hostname>
    User <your-user>
```

Once configured, you can simply run:
```bash
ssh uva
ssh spark
```

### SSH Agent Configuration
To avoid interactive password prompts when the tool automatically creates SSH tunnels for the backend, you should add your private key to the SSH agent:

```bash
# Start the ssh-agent in the background
eval "$(ssh-agent -s)"

# Add your private key to the agent
ssh-add ~/.ssh/id_rsa
```

Replace `~/.ssh/id_rsa` with the path to your actual private key if it's named differently.

To avoid running these commands every time you open a new terminal, you can add them to your shell configuration file (`~/.bashrc` for Bash or `~/.zshrc` for Zsh).

Add the following lines to the end of your `.bashrc` or `.zshrc`:

```bash
# Auto-start SSH agent and add key
if [ -z "$SSH_AUTH_SOCK" ]; then
   eval "$(ssh-agent -s)"
fi
ssh-add ~/.ssh/id_rsa 2>/dev/null
```

Ensure the path `~/.ssh/id_rsa` matches your actual private key file.
