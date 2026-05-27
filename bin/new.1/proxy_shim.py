from flask import Flask, request, Response
import requests
import json

app = Flask(__name__)
TARGET_URL = "https://localhost:8080"  # Your SSH Tunnel


@app.route("/<path:path>", methods=["POST"])
def proxy(path):
    # 1. Capture the original data from Cline
    data = request.get_json()

    # 2. STRIP problematic parameters that cause 422 errors
    # We only keep what the UVA Open WebUI specifically wants
    clean_payload = {
        "model": data.get("model", "Kimi K2.5"),
        "messages": data.get("messages", []),
        "stream": data.get("stream", True),
    }

    # 3. Prepare headers
    headers = {
        "Host": "open-webui.rc.virginia.edu",
        "Authorization": request.headers.get("Authorization"),
        "Content-Type": "application/json",
    }

    # 4. Forward to the tunnel
    try:
        resp = requests.post(
            f"{TARGET_URL}/{path}",
            headers=headers,
            json=clean_payload,  # Send only the sanitized payload
            verify=False,
            stream=True,
        )

        # 5. Return the response back to Cline
        return Response(
            resp.iter_content(chunk_size=1024),
            status=resp.status_code,
            headers=dict(resp.headers),
        )
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(port=8081)
