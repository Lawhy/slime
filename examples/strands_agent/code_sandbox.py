"""
Code sandbox module for executing code in a sandbox environment using `llm_sandbox`.
"""

from llm_sandbox.docker import SandboxDockerSession
from strands import tool


class CodeSandbox:
    def __init__(self, workdir: str = "code_sandbox", language: str = "python", execution_timeout: int = 300):
        self.session = SandboxDockerSession(
            workdir=workdir,
            lang=language,
            verbose=True,
            keep_template=True,
            execution_timeout=execution_timeout,
        )
        self.start_session()

    @tool
    def execute_code(self, code: str) -> str:
        """Execute the Python code in the sandbox environment."""
        result = self.session.run(code)
        return result.to_json()

    def start_session(self) -> None:
        """Start the session."""
        self.session.__enter__()

    def close_session(self) -> None:
        """Close the session."""
        self.session.__exit__(None, None, None)
