#!/usr/bin/env python3
"""UVA Kimi K2.5 Proxy Shim for LiteLLM Integration.

This proxy sits between LiteLLM and the UVA SSH tunnel, handling:
- Host header injection (required by UVA)
- Payload cleaning (removes params UVA doesn't support)
- SSL verification bypass (tunnel uses self-signed cert)
- Both streaming and non-streaming responses
- OpenAI-compatible /v1/models endpoint
"""

import json
import logging
import os
import sys
from pathlib import Path

import requests
from flask import Flask, Response, request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
TARGET_URL = os.environ.get("UVA_TARGET_URL", "https://localhost:8080")
UVA_HOST = "open-webui.rc.virginia.edu"


def get_uva_key() -> str:
    """Read UVA Kimi key from file or environment."""
    # First try environment variable
    key = os.environ.get("UVA_KIMI_KEY", "")
    if key:
        return key
    
    # Then try file
    key_path = Path.home() / "gemma" / "uva-kimmi-key.txt"
    try:
        if key_path.exists():
            return key_path.read_text().strip()
    except (IOError, PermissionError) as e:
        logger.error(f"Failed to read key file: {e}")
    
    return ""


def clean_payload(data: dict) -> dict:
    """Clean OpenAI payload for UVA compatibility.
    
    UVA's Open WebUI only supports basic parameters.
    """
    clean = {
        "model": data.get("model", "Kimi K2.5"),
        "messages": data.get("messages", []),
        "stream": data.get("stream", True),
    }
    
    # Only add optional params if present and supported
    if "temperature" in data:
        clean["temperature"] = data["temperature"]
    if "max_tokens" in data:
        clean["max_tokens"] = data["max_tokens"]
    
    return clean


@app.route("/v1/models", methods=["GET"])
def list_models():
    """OpenAI-compatible models endpoint."""
    return json.dumps({
        "object": "list",
        "data": [
            {
                "id": "Kimi K2.5",
                "object": "model",
                "created": 1700000000,
                "owned_by": "uva",
            }
        ]
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Proxy chat completions to UVA endpoint."""
    try:
        # 1. Get and clean payload
        data = request.get_json() or {}
        clean_data = clean_payload(data)
        
        # 2. Get authorization (from header or file)
        auth_header = request.headers.get("Authorization", "")
        if not auth_header or auth_header == "Bearer dummy-key":
            key = get_uva_key()
            if key:
                auth_header = f"Bearer {key}"
            else:
                logger.error("No UVA key available")
                return json.dumps({"error": "No authorization"}), 401
        
        # 3. Prepare headers
        headers = {
            "Host": UVA_HOST,
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }
        
        logger.info(f"Proxying request to {UVA_HOST}")
        logger.debug(f"Payload: {json.dumps(clean_data)[:200]}...")
        
        # 4. Forward to tunnel
        resp = requests.post(
            f"{TARGET_URL}/api/chat/completions",
            headers=headers,
            json=clean_data,
            verify=False,  # SSH tunnel uses self-signed cert
            stream=clean_data.get("stream", True),
            timeout=300,
        )
        
        # 5. Log 400 errors with request details for debugging
        if resp.status_code == 400:
            # Read response content (for non-streaming or if available)
            try:
                error_content = resp.text[:2000] if resp.text else "No response body"
            except Exception:
                error_content = "Unable to read response body"
            
            logger.error("=" * 80)
            logger.error("400 BAD REQUEST ERROR")
            logger.error("-" * 80)
            logger.error(f"Target URL: {TARGET_URL}/api/chat/completions")
            logger.error(f"Request Headers: {json.dumps(headers, indent=2)}")
            logger.error(f"Request Payload: {json.dumps(clean_data, indent=2)}")
            logger.error(f"Response Status: {resp.status_code}")
            logger.error(f"Response Body: {error_content}")
            logger.error("=" * 80)
        
        # Also log other error status codes
        elif resp.status_code >= 400:
            try:
                error_content = resp.text[:500] if resp.text else "No response body"
            except Exception:
                error_content = "Unable to read response body"
            logger.warning(f"HTTP {resp.status_code} from upstream: {error_content}")
        
        # 6. Stream or return full response
        if clean_data.get("stream", True):
            return Response(
                resp.iter_content(chunk_size=1024),
                status=resp.status_code,
                headers={
                    "Content-Type": resp.headers.get("Content-Type", "text/event-stream"),
                },
            )
        else:
            return Response(
                resp.content,
                status=resp.status_code,
                headers={"Content-Type": "application/json"},
            )
            
    except requests.exceptions.Timeout:
        logger.error("Request to UVA timed out")
        return json.dumps({"error": "Request timeout"}), 504
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return json.dumps({"error": f"Cannot connect to tunnel: {e}"}), 502
    except Exception as e:
        logger.exception("Unexpected error")
        return json.dumps({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return json.dumps({"status": "healthy", "target": TARGET_URL})


if __name__ == "__main__":
    # Disable SSL warnings for self-signed tunnel cert
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Check for key
    if not get_uva_key():
        logger.warning("No UVA key found. Set UVA_KIMI_KEY or create ~/gemma/uva-kimmi-key.txt")
    
    port = int(os.environ.get("PROXY_PORT", "8081"))
    logger.info(f"Starting proxy on port {port} -> {TARGET_URL}")
    app.run(host="127.0.0.1", port=port, debug=False)