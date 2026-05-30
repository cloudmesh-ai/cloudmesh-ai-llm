# The Proxy Service

The Cloudmesh AI Proxy Service acts as a centralized gateway between your client tools (like Cline or Aider) and multiple distributed vLLM backends.

## Introduction to the Proxy Service

Instead of managing separate SSH tunnels and API endpoints for every single GPU node, the proxy service provides a single, stable entry point. 

### Why use the Proxy Service?
- **Centralized Management**: One API key and one base URL for all your models.
- **Simplified Client Config**: You don't need to restart your client tools when switching from one remote backend to another.
- **Load Balancing & Routing**: The proxy can route requests to the most appropriate backend based on the model requested.
- **Security**: Encapsulates the complexity of the backend infrastructure.

### Architecture
The request flow follows this path:
`Client Tool (Cline/Aider)` $\rightarrow$ `Cloudmesh AI Proxy` $\rightarrow$ `vLLM Backend (Remote GPU Node)`

## Using Backends via Proxy

The proxy service uses the model name in the API request to determine which backend to route to.

### Configuring the Proxy
In your `llm.yaml`, you define the proxy endpoint as your primary base URL:

```yaml
cloudmesh:
  ai:
    proxy:
      base_url: "http://proxy.cloudmesh.ai/v1"
      api_key: "your-proxy-api-key"
```

### Step-by-Step: Accessing uva.gemma via Proxy
1. **Ensure the Backend is Running**: Start the service as described in the [Starting AI Services](./starting-services.md) guide.
2. **Configure Client**: Set your client's base URL to the proxy address.
3. **Request Model**: When you send a request for `google/gemma-2b-it`, the proxy identifies the active `uva.gemma` backend and forwards the request.

## Accessing Special Models: uva kimmi

`uva kimmi` is a specialized deployment optimized for specific research tasks. Accessing it via the proxy is the recommended method.

### Using uva kimmi
To use Kimmi, simply specify the model name in your client configuration or prompt:

**Model Name**: `kimmi` (or as specified in your proxy routing table)

The proxy handles the specific routing and authentication required for the Kimmi backend, providing a seamless OpenAI-compatible interface.