"""
Minimal Strands-Agents-based coding agent that uses the `CodeSandbox` to execute code in a sandbox environment.
"""


from code_sandbox import CodeSandbox
from strands import Agent
from strands.models import BedrockModel


# The default model is `claude-4.0`
model = BedrockModel()

code_sandbox = CodeSandbox(
    workdir="/workspace/examples/strands_agent/code_sandbox",
    language="python",
    execution_timeout=300,
)
code_sandbox.session.install(["numpy"])

if __name__ == "__main__":
    agent = Agent(model=model, tools=[code_sandbox.execute_code])
    agent(
        "Write a simple numpy program to calculate the eigen-values of a matrix and execute it on a random matrix."
    )
    code_sandbox.close_session()
    print("Code sandbox closed")
