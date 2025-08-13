# import os, re, subprocess, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from entity.code_quality import CodeQuality

# class AgentEvaluator:
#     """
#     Executes the final code with real data and parses AUROC/AUPRC.
#     (Logic ported from the old Reviewer.execute_code)
#     """

#     # ---------- public ----------
#     def execute_code(self, code: str, algorithm_name: str) -> CodeQuality:
#         # Create folder for generated scripts if it doesn't exist
#         folder = "./generated_scripts"
#         os.makedirs(folder, exist_ok=True)

#         # Save the provided code to a Python file
#         path = os.path.join(folder, f"{algorithm_name}.py")
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(code)

#         # Execute the script using subprocess and capture output
#         res = subprocess.run(["python", path], capture_output=True, text=True)
#         print("\n=== Real-Data Execution Output ===\n", res.stdout, res.stderr)

#         # If execution failed, return error result
#         if res.returncode != 0:
#             return CodeQuality(
#                 code=code, algorithm=algorithm_name, parameters={}, std_output="",
#                 error_message=res.stderr,
#                 auroc=-1, auprc=-1, error_points=[], review_count=0
#             )

#         # Parse metrics from the script output
#         auroc  = self._find_float(r"AUROC:\s*([\d.]+)", res.stdout)
#         auprc  = self._find_float(r"AUPRC:\s*([\d.]+)", res.stdout)
#         errors = self._parse_errors(res.stdout)

#         # Return evaluation result
#         return CodeQuality(
#             code=code, algorithm=algorithm_name, parameters={}, std_output=res.stdout,
#             error_message="", auroc=auroc, auprc=auprc,
#             error_points=errors, review_count=0
#         )

#     # ---------- helpers ----------
#     @staticmethod
#     def _find_float(pattern: str, text: str, default: float = -1.0) -> float:
#         # Find a float value in the text using regex
#         m = re.search(pattern, text)
#         return float(m.group(1)) if m else default

#     @staticmethod
#     def _parse_errors(text: str):
#         # Extract prediction failure points from the text
#         pts = []
#         for line in text.splitlines():
#             if "Failed prediction at point" in line:
#                 m = re.search(r"\[([^\]]+)] with true label ([\d.]+)", line)
#                 if m:
#                     nums = [float(x.strip()) for x in m.group(1).split(",")]
#                     pts.append({"point": nums, "true_label": float(m.group(2))})
#         return pts
import os, re, subprocess, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

from entity.code_quality import CodeQuality

def print_python_code(code_str):
    """Pretty-print Python code in the terminal."""
    print(highlight(code_str, PythonLexer(), TerminalFormatter()))

class AgentEvaluator:
    """
    Executes the final code with real data and parses AUROC/AUPRC.
    (Logic ported from the old Reviewer.execute_code)
    """

    # ---------- public ----------
    def execute_code(self, code: str, algorithm_name: str) -> CodeQuality:
        # Clean up code first
        cleaned_code = self._clean_markdown(code)

        print("\n[DEBUG] Final Cleaned Code to Execute:\n")
        print_python_code(cleaned_code)

        # Create folder for generated scripts if it doesn't exist
        folder = "./generated_scripts"
        os.makedirs(folder, exist_ok=True)

        # Save the cleaned code to a Python file
        path = os.path.join(folder, f"{algorithm_name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned_code)

        # Execute the script using subprocess and capture output
        res = subprocess.run(["python", path], capture_output=True, text=True)
        print("\n=== Real-Data Execution Output ===\n", res.stdout, res.stderr)

        # If execution failed, return error result
        if res.returncode != 0:
            return CodeQuality(
                code=cleaned_code, algorithm=algorithm_name, parameters={}, std_output="",
                error_message=res.stderr,
                auroc=-1, auprc=-1, error_points=[], review_count=0
            )

        # Parse metrics from the script output
        auroc  = self._find_float(r"AUROC:\s*([\d.]+)", res.stdout)
        auprc  = self._find_float(r"AUPRC:\s*([\d.]+)", res.stdout)
        errors = self._parse_errors(res.stdout)

        # Return evaluation result
        return CodeQuality(
            code=cleaned_code, algorithm=algorithm_name, parameters={}, std_output=res.stdout,
            error_message="", auroc=auroc, auprc=auprc,
            error_points=errors, review_count=0
        )

    # ---------- helpers ----------
    @staticmethod
    def _clean_markdown(txt: str) -> str:
        """Remove markdown code fences from the script."""
        txt = re.sub(r"```(python)?", "", txt)
        return re.sub(r"```", "", txt).strip()

    @staticmethod
    def _find_float(pattern: str, text: str, default: float = -1.0) -> float:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    @staticmethod
    def _parse_errors(text: str):
        pts = []
        for line in text.splitlines():
            if "Failed prediction at point" in line:
                m = re.search(r"\[([^\]]+)] with true label ([\d.]+)", line)
                if m:
                    nums = [float(x.strip()) for x in m.group(1).split(",")]
                    pts.append({"point": nums, "true_label": float(m.group(2))})
        return pts
