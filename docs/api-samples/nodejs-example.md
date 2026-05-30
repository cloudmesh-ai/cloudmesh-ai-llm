# Node.js Integration Example

This guide demonstrates how to connect a Node.js application to the vLLM backend via the local tunnel provided by the orchestrator.

## Implementation Example

We recommend using `axios` for simple HTTP requests or the official `openai` Node.js library for full compatibility.

### Using Axios (Lightweight)

```javascript
const axios = require('axios');

async function chatWithModel(prompt) {
    const url = 'http://localhost:8000/v1/chat/completions';
    const apiKey = process.env.VLLM_API_KEY || 'token-not-needed-for-local';

    const data = {
        model: 'google/gemma-4-31B-it', // Replace with your loaded model
        messages: [
            { role: 'system', content: 'You are a helpful AI assistant.' },
            { role: 'user', content: prompt }
        ],
        temperature: 0.7,
        max_tokens: 1024
    };

    try {
        const response = await axios.post(url, data, {
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            }
        });

        console.log('Assistant:', response.data.choices[0].message.content);
    } catch (error) {
        if (error.code === 'ECONNREFUSED') {
            console.error('Connection Error: Is the tunnel active? Run `cmc llm status` to check.');
        } else {
            console.error('API Error:', error.response ? error.response.data : error.message);
        }
    }
}

// Usage
chatWithModel('Explain the difference between a GPU node and a CPU node in an HPC cluster.');
```

### Using the OpenAI SDK (Recommended)

```javascript
const OpenAI = require('openai');

const openai = new OpenAI({
    baseURL: 'http://localhost:8000/v1',
    apiKey: process.env.VLLM_API_KEY || 'token-not-needed-for-local',
});

async function main() {
    try {
        const completion = await openai.chat.completions.create({
            messages: [{ role: 'user', content: 'Hello!' }],
            model: 'google/gemma-4-31B-it',
        });

        console.log(completion.choices[0].message.content);
    } catch (e) {
        console.error('Error:', e);
    }
}

main();
```

## Key Integration Tips for Node.js

### 1. Environment Variables
Use the `dotenv` package to manage your API keys and ports securely:
```bash
npm install dotenv
```
```javascript
require('dotenv').config();
const port = process.env.CLOUDMESH_AI_PORT || 8000;
```

### 2. Handling Streams
If you need real-time token streaming, use the `openai` SDK with `stream: true`. This prevents the application from appearing "frozen" during long generations.

### 3. Timeout Management
Remote GPU nodes can sometimes have higher latency during the first few tokens. Increase the timeout in your HTTP client:
```javascript
const response = await axios.post(url, data, { timeout: 30000 }); // 30 seconds