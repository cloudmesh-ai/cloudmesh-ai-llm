# UVA Kimi K2.5 - Cline Local Setup Guide

This guide walks you through setting up a local connection from Cline/VS Code to the University of Virginia's Kimi K2.5 model via the Open WebUI infrastructure.

## Overview

The UVA Kimi bridge connects your local Cline extension to UVA's hosted Kimi K2.5 model. The architecture uses an SSH tunnel for secure connectivity and a lightweight Flask proxy (`proxy_shim.py`) to sanitize requests and handle streaming responses.

```
┌─────────────┐     HTTP      ┌──────────────┐     SSH Tunnel     ┌─────────────────────┐
│   Cline     │ ─────────────> │ proxy_shim   │ ─────────────────> │ UVA Open WebUI      │
│  VS Code    │   Port 8081   │   Port 8080  │    Port 8080       │  open-webui.rc.     │
│             │ <───────────── │              │ <───────────────── │  virginia.edu:443   │
└─────────────┘   Streaming    └──────────────┘                    └─────────────────────┘
```

## Prerequisites

Before starting, ensure you have:

- **SSH access to UVA** with key-based authentication configured
- **API Key** stored at `~/gemma/uva-kimmi-key.txt` (see [Obtaining Your API Key](#obtaining-your-api-key) below)
- **Python 3** with dependencies installed:
  ```bash
  pip install -r requirements.txt
  ```

  Or install individually:
  ```bash
  pip install flask requests pyyaml tabulate
  ```

### UVA Open WebUI Portal

The UVA Kimi interface is available at:

**https://open-webui.rc.virginia.edu**

This web interface provides direct access to the Kimi K2.5 model through your browser, and is also where you generate your API key.

### Obtaining Your API Key

To use Cline with UVA Kimi, you need an API key from the Open WebUI portal:

1. **Visit** https://open-webui.rc.virginia.edu in your browser
2. **Sign in** with your UVA credentials
3. **Click your profile** (top-right corner) → **Settings**
4. **Navigate to** **Account** → **API Key**
5. **Generate a new API key** and copy it
6. **Save the key** to `~/gemma/uva-kimmi-key.txt`:
   ```bash
   mkdir -p ~/gemma
   echo "your_api_key_here" > ~/gemma/uva-kimmi-key.txt
   chmod 600 ~/gemma/uva-kimmi-key.txt
   ```

> **Important**: Keep your API key secure. Do not commit it to version control or share it.

## Step 1: SSH Configuration

### Option A: Quick SSH Command

Open a dedicated terminal and run:

```bash
ssh -L 8080:open-webui.rc.virginia.edu:443 uva
```

Keep this terminal open - the tunnel must remain active for requests to work.

### Option B: SSH Config (Recommended)

Add to your `~/.ssh/config`:

```
Host uva-genai
    HostName open-webui.rc.virginia.edu
    User your_uva_username
    LocalForward 8080 open-webui.rc.virginia.edu:443
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Then connect with:
```bash
ssh -N uva-genai
```

## Step 2: Start the Proxy Shim

The `proxy_shim.py` script is required because Cline sends parameters that the UVA Open WebUI rejects (causing 422 errors). The shim strips problematic fields and forwards only the essential payload.

### Running the Proxy

```bash
cd new
python proxy_shim.py
```

The proxy will start on **port 8081** and output:
```
 * Running on http://127.0.0.1:8081
```

### What the Proxy Does

The `proxy_shim.py` performs these critical functions:

1. **Receives** Cline's requests on `http://localhost:8081`
2. **Sanitizes** the payload by keeping only:
   - `model`: "Kimi K2.5"
   - `messages`: The conversation messages
   - `stream`: Streaming flag (usually `true`)
3. **Adds required headers** including the `Host: open-webui.rc.virginia.edu` header
4. **Forwards** to the SSH tunnel at `https://localhost:8080`
5. **Streams** responses back to Cline

## Step 3: Configure Cline in VS Code

1. **Open Cline Settings** (click the Cline icon → Settings/Configure API)

2. **Select Provider**: `OpenAI Compatible`

3. **Configure these settings**:

   | Setting | Value |
   |---------|-------|
   | **Base URL** | `http://localhost:8081` |
   | **API Key** | Contents of `~/gemma/uva-kimmi-key.txt` |
   | **Model ID** | `Kimi K2.5` |

4. **Save and Test**

## Step 4: Test the Connection

### Quick Test with curl

```bash
curl -X POST http://localhost:8081/api/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(cat ~/gemma/uva-kimmi-key.txt)" \
  -d '{
    "model": "Kimi K2.5",
    "messages": [{"role": "user", "content": "Hello, are you working?"}],
    "stream": false
  }'
```

### Test with Python Script

Use the provided `kimitest.py` for a readable streaming test:

```bash
cd new
python kimitest.py
```

This will display the streaming response in real-time.

### Expected Response

You should see a JSON response with the model's reply. If streaming is enabled, you'll receive SSE (Server-Sent Events) formatted chunks.

## File Reference

| File | Purpose | Location |
|------|---------|----------|
| `proxy_shim.py` | Flask proxy that sanitizes Cline requests | `new/proxy_shim.py` |
| `kimitest.py` | Test script for API verification | `new/kimitest.py` |
| `uva-kimmi-key.txt` | Your API key | `~/gemma/uva-kimmi-key.txt` |

## Troubleshooting

### Connection Refused Error

**Symptom**: `curl: (7) Failed to connect to localhost port 8081`

**Fix**: Ensure the proxy is running:
```bash
python new/proxy_shim.py
```

### 401 Unauthorized

**Symptom**: API returns authentication error

**Fix**: 
- Verify your API key file exists: `cat ~/gemma/uva-kimmi-key.txt`
- Check for extra whitespace: `tr -d '[:space:]' < ~/gemma/uva-kimmi-key.txt`
- Ensure the key has not expired

### 422 Unprocessable Entity (Without Proxy)

**Symptom**: Direct connection to SSH tunnel fails

**Fix**: Always use the proxy. Direct connections to `https://localhost:8080` will fail because Cline sends parameters the UVA endpoint doesn't accept. Route through `http://localhost:8081`.

### SSL Certificate Errors

**Symptom**: SSL verification errors when testing with curl

**Fix**: Use `-k` flag with curl (the proxy already disables verification for the tunnel):
```bash
curl -k https://localhost:8080/...
```

### SSH Tunnel Closed

**Symptom**: All requests suddenly fail

**Fix**: The SSH tunnel may have closed. Re-establish it:
```bash
ssh -L 8080:open-webui.rc.virginia.edu:443 uva
```

### Model Not Found

**Symptom**: "Model not found" or similar error

**Fix**: Ensure exact model name in Cline: `Kimi K2.5` (case-sensitive)

## Automation Tips

### Bash Function for Quick Queries

Add to `~/.bashrc` or `~/.zshrc`:

```bash
ask_uva() {
    curl -s -X POST "http://localhost:8081/api/chat/completions" \
         -H "Authorization: Bearer $(tr -d '[:space:]' < ~/gemma/uva-kimmi-key.txt)" \
         -H "Content-Type: application/json" \
         -d "{\"model\": \"Kimi K2.5\", \"messages\": [{\"role\": \"user\", \"content\": \"$1\"}], \"stream\": false}" \
    | jq -r '.choices[0].message.content'
}
```

Usage:
```bash
ask_uva "Explain quantum computing"
```

### Systemd/User Service (Advanced)

For persistent tunnel, create `~/.config/systemd/user/uva-genai.service`:

```ini
[Unit]
Description=UVA GenAI SSH Tunnel
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/ssh -NT -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes -L 8080:open-webui.rc.virginia.edu:443 uva
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

Enable and start:
```bash
systemctl --user enable uva-genai
systemctl --user start uva-genai
```

## Architecture Notes

### Why the Proxy is Necessary

The UVA Open WebUI endpoint has a strict API schema. Cline (and many OpenAI-compatible clients) send additional parameters like:

- `temperature`
- `top_p`
- `presence_penalty`
- `frequency_penalty`
- `tools` / `tool_choice`
- `response_format`

These cause **422 Unprocessable Entity** errors. The `proxy_shim.py` strips these and forwards only the essential fields that UVA accepts.

### Security Considerations

- The SSH tunnel encrypts all traffic between your machine and UVA
- The proxy runs locally only (binds to `127.0.0.1`)
- API keys are never logged or transmitted beyond the UVA endpoint
- SSL verification is disabled for the local tunnel leg (localhost) only; the SSH tunnel itself is encrypted

## Quick Command Reference

```bash
# 1. Start SSH tunnel (Terminal 1)
ssh -L 8080:open-webui.rc.virginia.edu:443 uva

# 2. Start proxy (Terminal 2)
cd new && python proxy_shim.py

# 3. Test connection
python new/kimitest.py

# 4. Configure Cline in VS Code
#    Base URL: http://localhost:8081
#    API Key: (from ~/gemma/uva-kimmi-key.txt)
#    Model: Kimi K2.5
```

## Related Documentation

- `new/kimi.md` - Original UVA bridge manual
- `new/README.md` - Local LLM infrastructure guide
- `new/proxy_shim.py` - Proxy implementation source