
import os, re, subprocess, sys, ast, importlib.util
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
    Also auto-installs missing third-party libs in the *current* Python env (your venv)
    before executing the script.
    """

    # ---------- public ----------
    def execute_code(self, code: str, algorithm_name: str) -> CodeQuality:
        # 1) Clean code (strip markdown fences)
        cleaned_code = self._clean_markdown(code)

        print("\n[DEBUG] Final Cleaned Code to Execute:\n")
        print_python_code(cleaned_code)

        # 2) Ensure output folder
        folder = "./generated_scripts"
        os.makedirs(folder, exist_ok=True)

        # 3) Save the code to a file
        path = os.path.join(folder, f"{algorithm_name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned_code)

        # Execute the script using subprocess and capture output
        # res = subprocess.run(["python", path], capture_output=True, text=True)
        res = subprocess.run([sys.executable, path], capture_output=True, text=True)

        print("\n=== Real-Data Execution Output ===\n", res.stdout, res.stderr)

        # 6) If execution failed, return the error
        if res.returncode != 0:
            return CodeQuality(
                code=cleaned_code, algorithm=algorithm_name, parameters={}, std_output=res.stdout,
                error_message=res.stderr, auroc=-1, auprc=-1, error_points=[], review_count=0
            )

        # 7) Parse metrics from the script output
        auroc  = self._find_float(r"AUROC:\s*([\d.]+)", res.stdout)
        auprc  = self._find_float(r"AUPRC:\s*([\d.]+)", res.stdout)
        errors = self._parse_errors(res.stdout)

        # 8) Return evaluation result
        return CodeQuality(
            code=cleaned_code, algorithm=algorithm_name, parameters={}, std_output=res.stdout,
            error_message="", auroc=auroc, auprc=auprc,
            error_points=errors, review_count=0
        )

    # ---------- deps handling ----------
    def _ensure_dependencies(self, code_str: str) -> None:
        """
        Parse imports from code, detect which top-level modules are missing in the
        current environment, and pip-install only the missing ones using sys.executable.
        """
        modules = self._discover_imports(code_str)
        if not modules:
            return

        # Map top-level module -> pip package name
        module_to_pip = {
            # core DS/ML
            "numpy": "numpy",
            "pandas": "pandas",
            "scipy": "scipy",
            "sklearn": "scikit-learn",
            "statsmodels": "statsmodels",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            # anomaly/time-series libs
            "pyod": "pyod",
            "pygod": "pygod",
            "darts": "u8darts",
            # DL
            "torch": "torch",
            "torchvision": "torchvision",
            "tensorflow": "tensorflow",
            "pytorch_lightning": "pytorch-lightning",
            # CV / utils
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "skimage": "scikit-image",
            # others often used
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "catboost": "catboost",
        }

        # stdlib modules to ignore
        stdlib = set(getattr(sys, "stdlib_module_names", set())) or {
            "os","sys","time","math","random","datetime","re","json","subprocess",
            "itertools","functools","collections","typing","pathlib","csv","ast"
        }

        to_install = []
        for m in sorted(modules):
            if m in stdlib:
                continue
            # ignore relative or local imports (handled in _discover_imports)
            if importlib.util.find_spec(m) is None:
                to_install.append(module_to_pip.get(m, m))

        if not to_install:
            print("[INFO] All required third-party packages already available.")
            return

        print(f"[INFO] Installing missing packages: {to_install}")
        # install in one go; raises CalledProcessError on failure
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])

    def _discover_imports(self, code_str: str) -> set:
        """
        Robustly discover top-level modules via AST.
        Returns a set of names like {'numpy','pandas','sklearn',...}
        """
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            # fallback to regex if code has minor syntax noise
            return set(re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", code_str, re.MULTILINE))

        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = (alias.name or "").split(".")[0]
                    if root:
                        mods.add(root)
            elif isinstance(node, ast.ImportFrom):
                # skip relative imports (node.level > 0)
                if node.level and node.level > 0:
                    continue
                if node.module:
                    root = node.module.split(".")[0]
                    if root:
                        mods.add(root)
        return mods

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
