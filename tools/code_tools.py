# tools/code_tools.py
from langchain_core.tools import tool
from e2b_code_interpreter import CodeInterpreter
import os


@tool
def python_repl(code: str) -> str:
    """
    Executes Python code in a secure Jupyter sandbox.
    You can use print() to output results. You can also pip install libraries. 
    """
    print(f"\n[Executing Code in E2B Sandbox]:\n{code}\n")

    try:
        with CodeInterpreter() as sandbox:
            # Send the code to the E2B micro-VM
            execution = sandbox.notebook.exec_cell(code)


        # If there are errors, return them 
        if execution.error:
            return f"Execution Error: {execution.error.name}: {execution.error.value}\n{execution.error.traceback}"

        # Return standard output
        output_str = ""
        if execution.logs.stdout:
            output_str += "\n".join(execution.logs.stdout)


        # If the LLM generated a chart or dataframe, it's stored in results
        if execution.results:
            for result in execution.results:
                if result.text:
                    output_str += f"\n[Result]: {result.text}"

        if not output_str.strip():
            return "Executed successfully with no output. Did you forget to print?"

        return output_str

    except Exception as e:
        return f"System Error connecting to sandbox: {e}"
