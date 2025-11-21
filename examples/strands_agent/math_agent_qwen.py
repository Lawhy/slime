"""
Minimal Strands-Agents-based math agent that uses `calculator` tool (`sympy`-based).

To start the vLLM server:

docker run --gpus all \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --model Qwen/Qwen3-0.6B
"""

import json
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

model = OpenAIModel(
    client_args={
        "api_key": "EMPTY",
        "base_url": "http://localhost:8000/v1",
    },
    model_id="Qwen/Qwen3-0.6B",
    params={
        "max_tokens": 2048,
        "temperature": 1.0,
    },
)

if __name__ == "__main__":
    agent = Agent(model=model, tools=[calculator])
    agent("What is the square root of 16?")
    print("\n\n-------------View OpenAI-formatted Messages------------------")
    openai_messages = agent.model.format_request_messages(messages=agent.messages, system_prompt=agent.system_prompt)
    for message in openai_messages:
        message["content"] = message["content"][0]["text"]
    print(
        json.dumps(
            openai_messages, indent=2
        )
    )
