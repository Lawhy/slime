"""
Code sandbox module for executing code in a sandbox environment using `llm_sandbox`.
"""

from llm_sandbox.docker import SandboxDockerSession
from strands import tool


class CodeSandbox:
    def __init__(self, workdir: str = "code_sandbox", language: str = "python", execution_timeout: int = 300):
        self.workdir = workdir
        self.language = language
        self.execution_timeout = execution_timeout

    @tool
    def execute_python_code(self, code: str, libraries: list[str] = None) -> str:
        """
        Execute Python code in an isolated Docker sandbox environment with optional dependency installation.
        
        Args:
            code (str): The Python code to execute
            libraries (list[str], optional): A list of Python libraries to install and use, e.g. ["numpy"]
            
        Returns:
            A JSON string containing the execution result with:
            - exit_code: Exit code of the execution (0 for success)
            - stderr: Error messages if any
            
        Example:
            code = "import numpy as np\nprint(np.random.rand())"
            
        Note: The sandbox has a timeout limit and runs in an isolated environment.
        """
        with SandboxDockerSession(
            workdir=self.workdir,
            lang=self.language,
            verbose=True,
            keep_template=True,
            execution_timeout=self.execution_timeout,
        ) as session:
            result = session.run(code=code, libraries=libraries)
            return result.to_json()
