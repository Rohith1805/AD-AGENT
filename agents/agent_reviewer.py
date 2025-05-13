import subprocess, os, re, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entity.code_quality import CodeQuality
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Initialize the OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Prompt template for generating synthetic test data
test_prompt = PromptTemplate.from_template("""
You will receive a Python script for {package_name} that trains an anomaly‑detection model with real datasets.

--- BEGIN CODE ---
{code}
--- END CODE ---
                                           
TASK:
1. Replace **all data‑loading operations** (DataLoader, torch.load, np.load, pandas.read*, etc.)
   with code that creates SMALL synthetic data directly in the script:
   • For PyOD: generate X_train, y_train, X_test, y_test using `generate_data`; 
     `from pyod.utils.data import generate_data`
     `X_train, X_test, y_train, y_test = generate_data(n_train=200, n_test=100, contamination=0.1)`
   • For PyGOD: build train and test graph follow instruction below;
     `import torch`
     `from pygod.generator import gen_contextual_outlier, gen_structural_outlier`
     `from torch_geometric.data import Data`
     `num_nodes = 200`  
     `num_features = 16`  
     `x = torch.randn(num_nodes, num_features)`  

     `edge_index = torch.tensor([`  
     `    [i, (i+1) % num_nodes] for i in range(num_nodes)`  
     `], dtype=torch.long).T  # shape: [2, num_edges]`  

     `data = Data(x=x, edge_index=edge_index)`  
     `data, ya = gen_contextual_outlier(data, n=100, k=50)`  
     `data, ys = gen_structural_outlier(data, m=10, n=10)`  
     `data.y = torch.logical_or(ys, ya).long()`  
   • For tslib:
     Do not generate any new code. 
     1. Just change the value of `--data` parameter to `MSL` and `--root_path` to `./data/unit_test`. Since I want to run unit test on the data called `MSL` rahter than origial data
     2.     "--seq_len", "10",
            "--label_len", "5",
            "--pred_len", "0",
            "--train_epochs", "2"
            "--enc_in", "55"
        Set these three parameters to 10, 5, 0, 2, 55 respectively. This is for small data set unit test only
        You have to set `--enc_in` to 55 to match the data dimension. And chnage all dimention related parameters to 55
     
   • For Darts:

    `import numpy as np`
    `import pandas as pd`
    `from darts import TimeSeries`


    `def load_series(path: str,`
                    `n_samples: int = 500,`
                    `n_features: int = 1,`
                    `contamination: float = 0.05,`
                    `seed: int = 42):`
        `rng = np.random.default_rng(seed)`

        `dates = pd.date_range("2020-01-01", periods=n_samples, freq="H")`

        `data = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))`

        `n_anom = int(n_samples * contamination)`
        `anom_idx = rng.choice(n_samples, n_anom, replace=False)`
        `data[anom_idx] += rng.normal(loc=6.0, scale=1.0, size=(n_anom, n_features))`

        `df = pd.DataFrame(data, columns=[f"value_{{i+1}}" for i in range(n_features)])`
        `df["timestamp"] = dates`
        `df["anomaly"] = 0`
        `df.loc[anom_idx, "anomaly"] = 1`
        `df.set_index("timestamp", inplace=True)`

        `value_cols = [c for c in df.columns if c.startswith("value_")]`
        `series = TimeSeries.from_dataframe(df, value_cols=value_cols)`
        `labels = df["anomaly"].astype(int).values`
        `return series, labels`

    `series_train, y_train = load_series(None, n_samples=1000, n_features=3, seed=0)`
    `series_test,  y_test  = load_series(None, n_samples=300,  n_features=3, seed=1)`
    `series_train = series_train.astype(np.float32)`
    `series_test  = series_test.astype(np.float32)`
    `torch.set_default_dtype(torch.float32)`
                                           
    2. Keep the variable names and the rest of the logic unchanged.
    3. Output runnable Python **code only** (no explanations, no markdown).
""")

class AgentReviewer:
    """Responsible for executing code and recording metrics only."""
    def __init__(self):
        pass

    def test_code(
        self,
        code: str,
        algorithm_name: str,
        package_name: str
    ) -> str:
        """
        Generate a test script using synthetic data and execute it.
        Return an empty string on success, or an error message on failure or exception.
        """
        try:
            # 1) Use LLM to rewrite the script to use synthetic data
            test_script = llm.invoke(
                test_prompt.invoke({
                    "code": code,
                    "algorithm_name": algorithm_name,
                    "package_name": package_name
                })
            ).content
            test_script = self._clean_markdown(test_script)

            # 2) Save the rewritten script to file
            folder = "generated_scripts"
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{algorithm_name}_test.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(test_script)

            # 3) Execute the test script
            res = subprocess.run(["python", path],
                                 capture_output=True, text=True)
            print("\n=== Test Execution Output ===\n",
                  res.stdout, res.stderr)

            if res.returncode != 0:
                return res.stderr
            else:
                return ""
        except Exception as e:
            print(f"[test_code] Exception: {e}")
            return str(e)

    @staticmethod
    def _clean_markdown(txt: str) -> str:
        """Remove markdown code fences from the script."""
        txt = re.sub(r"```(python)?", "", txt)
        return re.sub(r"```", "", txt).strip()

    # -------- helpers --------
    @staticmethod
    def _find(pattern, text, default=-1.0):
        """Find a float number from text using regex pattern."""
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    @staticmethod
    def _find_errors(text):
        """Extract failed prediction points and true labels from output logs."""
        pts = []
        for line in text.splitlines():
            if "Failed prediction at point" in line:
                m = re.search(r"\[([^\]]+)] with true label ([\d.]+)", line)
                if m:
                    nums = [float(x.strip()) for x in m.group(1).split(",")]
                    pts.append({"point": nums, "true_label": float(m.group(2))})
        return pts
