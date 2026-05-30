# Advanced Python Integration

This guide shows how to integrate the vLLM orchestrator backends into a professional Python application, including streaming, error handling, and session management.

## Implementation Example

```python
import os
from openai import OpenAI
from openai import OpenAIError
import time

def create_vllm_client():
    """
    Initialize the OpenAI client to connect to the local tunnel.
    Assumes the server is running via `cmc llm start`.
    """
    return OpenAI(
        base_url="http://localhost:8000/v1", # Default vLLM tunnel port
        api_key=os.getenv("VLLM_API_KEY", "token-not-needed-for-local")
    )

def chat_with_model(prompt, stream=True):
    client = create_vllm_client()
    
    try:
        response = client.chat.completions.create(
            model="google/gemma-4-31B-it", # Replace with your loaded model
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=stream,
            temperature=0.7,
            max_tokens=1024
        )

        if stream:
            print("Assistant: ", end="")
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
            print("\n")
        else:
            print(f"Assistant: {response.choices[0].message.content}")

    except OpenAIError as e:
        print(f"API Error: {e}")
    except ConnectionError:
        print("Connection Error: Is the tunnel active? Run `cmc llm status` to check.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    user_prompt = "Explain the benefits of using a vLLM orchestrator for GPU clusters."
    print(f"User: {user_prompt}")
    chat_with_model(user_prompt)
```

## Key Integration Tips

### 1. Handling Tunnel Flapping
In production, tunnels can occasionally drop. Implement a retry mechanism with exponential backoff:
```python
import time
from openai import OpenAIError

def call_with_retry(func, retries=3):
    for i in range(retries):
        try:
            return func()
        except (OpenAIError, ConnectionError) as e:
            if i == retries - 1: raise e
            wait = (2 ** i)
            print(f"Connection lost. Retrying in {wait}s...")
            time.sleep(wait)
```

### 2. Model Versioning
Since the orchestrator can switch between `uva.gemma` and `uva.gemma2`, ensure your code dynamically resolves the model name from your `llm.yaml` config rather than hardcoding it.

### 3. Performance Tuning
For high-throughput applications, adjust the `max_tokens` and `temperature` based on the specific use case (e.g., lower temperature for extraction, higher for creative writing).