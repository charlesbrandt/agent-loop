import subprocess
from langchain.tools import BaseTool
from typing import Type, Any
from pydantic.v1 import BaseModel, Field

class ShellToolInput(BaseModel):
    command: str = Field(description="The shell command to execute.")

class ShellTool(BaseTool):
    name: str = "execute_shell_command"
    description: str = "Executes a shell command in a subprocess and returns the output."
    args_schema: Type[BaseModel] = ShellToolInput

    def _run(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=False  # Do not raise exception on non-zero exit codes
            )
            output = f"Return Code: {result.returncode}\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            return output
        except Exception as e:
            return f"An error occurred while executing the command: {e}"

    def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ShellTool does not support async execution.")
