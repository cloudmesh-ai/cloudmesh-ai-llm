import argparse, uvicorn, time
from fastapi import FastAPI, Header
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "google/gemma-4-31B-it", "object": "model"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, authorization: Optional[str] = Header(None)
):
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I am a mock Gemma 4 model. Your request was successful.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import os, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18123)
    args = parser.parse_args()

    pid = os.getpid()
    print(f"movLLM is up and running on {args.host}:{args.port}", flush=True)
    print(f"(APIServer pid={pid}) INFO:     Started server process [{pid}]", file=sys.stderr, flush=True)
    print(f"(APIServer pid={pid}) INFO:     Waiting for application startup.", file=sys.stderr, flush=True)
    print(f"(APIServer pid={pid}) INFO:     Application startup complete.", file=sys.stderr, flush=True)

    uvicorn.run(app, host=args.host, port=args.port)
