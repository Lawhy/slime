"""
Minimal Strands-Agents-based coding agent that uses the `CodeSandbox` to execute code in a sandbox environment.
"""

from strands import Agent
from strands.models.bedrock import BedrockModel

from code_sandbox import CodeSandbox

# Default to `Claude 4.0` model on AWS Bedrock
model = BedrockModel()

code_sandbox = CodeSandbox(
    workdir="/tmp/code_sandbox",
    language="python",
    execution_timeout=300,
)
code_sandbox.session.install(["numpy"])

if __name__ == "__main__":
    agent = Agent(model=model, tools=[code_sandbox.execute_code])
    agent(
        "Write a simple numpy program to calculate the eigen-values of a matrix and execute it on a "
        "simple matrix [[1, 2, 3], [4, 5, 6], [7, 8, 9]]."
    )
    code_sandbox.close_session()
    print("Code sandbox closed")
