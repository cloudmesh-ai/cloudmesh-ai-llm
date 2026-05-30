# API Sample Programs

Cloudmesh AI backends are fully compatible with the OpenAI API specification. This means you can use the official `openai` Python library to interact with your models.

## Python Client Examples

First, install the OpenAI library:

```bash
pip install openai
```

### Simple Chat Completion

This example shows how to send a prompt to a local or remote vLLM backend and receive a response.

```python
from openai import OpenAI

# Configuration
# Use http://localhost:8000/v1 for direct tunnel
# Use http://proxy.cloudmesh.ai/v1 for proxy service
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed-for-local" 
)

try:
    response = client.chat.completions.create(
        model="google/gemma-2b-it",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in one sentence."}
        ],
        temperature=0.7
    )
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print(f"Error connecting to vLLM: {e}")
```

### Streaming Responses

For a more interactive experience, use the streaming API to receive tokens as they are generated.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed-for-local"
)

stream = client.chat.completions.create(
    model="google/gemma-2b-it",
    messages=[{"role": "user", "content": "Write a short poem about GPUs."}],
    stream=True,
)

print("Streaming response: ", end="")
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

## Advanced Integration

When building production applications on top of remote vLLM instances, consider the following best practices:

### Handling Timeouts and Retries

Remote GPU nodes can sometimes experience transient network issues or high load. Implement a retry strategy:

```python
import time
from openai import OpenAI, APIConnectionError, APITimeoutError

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

def safe_completion(prompt, retries=3):
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model="google/gemma-2b-it",
                messages=[{"role": "user", "content": prompt}]
            )
        except (APIConnectionError, APITimeoutError) as e:
            if i == retries - 1:
                raise e
            print(f"Connection failed, retrying in {2**i}s...")
            time.sleep(2**i)

# Usage
res = safe_completion("Hello!")
```

### Context Window Management

vLLM allows for large context windows, but exceeding them will result in errors. Always monitor the token count of your input messages to ensure they fit within the model's limits (e.g., 8192 tokens for standard Gemma).