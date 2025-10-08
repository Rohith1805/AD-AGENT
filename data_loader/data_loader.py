# import google.generativeai as genai
# import os
# import re
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
# # import sklearn
# # from orion.data import load_signal

# import json 
# class DataLoader:
#     """
#     A class to load various file formats (.mat, .csv, .json, etc.) and extract data.
#     """

#     def __init__(self, filepath, desc='', store_script = False, store_path = 'generated_data_loader.py'):
#         """
#         Initialize DataLoader with the path to the file.
#         """
#         self.filepath = filepath
#         self.desc = desc

#         self.X_name = 'X'
#         self.y_name = 'y'


#         if not os.path.exists(self.filepath):
#             raise FileNotFoundError(f"File not found: {self.filepath}")
        
#         self.store_script = store_script
#         self.store_path = store_path
#         self.head = None
#     def generate_script_for_data_head(self):
#         file_path = self.filepath.replace("\\", "/")  # Normalize for cross-platform compatibility
#         file_type = self.filepath.split('.')[-1]  # Extract file extension

#         prompt = f"""
# Write a complete Python script for the given graph dataset:

# 1. **Required imports**: os, scipy.io, pandas, json, numpy. Add torch if the extension is `.pt`.
# 2. File path: "{file_path}" (already known). Do not ask for user input.
# 3. Immediately check if the file exists with `os.path.exists("{file_path}")`.  
#    If it does not exist, print a clear error message and exit.
# 4. Load the file based on its extension `{file_type}` without using if/elif for type detection in the script. 
# 5. Store the data in a variable called `X`.  
#    Set `y = "graph"` (always).
# 6. **MANDATORY**:
#    - At the end of the script, `X` and `y` **must** exist in `locals()` with exactly those names.
#    - Never leave `X` or `y` undefined.
#    - Ensure `X` is a valid object (NumPy array or loaded graph object).
# 7. Do **not** include any logic to guess dataset type — it is already given.

# Return only the Python code.
# """

#          # Initialize OpenAI client
#         # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         # client = genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyDzl7RXk0Gn6hlWHDiu5CkFoFfhLkv-D-c"))
#         genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#         model = genai.GenerativeModel("gemini-2.5-pro")
#         # Get response from GPT
#         response = model.generate_content(prompt)
     
#         if hasattr(response, "candidates") and response.candidates:
#             parts = response.candidates[0].content.parts
#             if parts:
#                 content = "".join(part.text for part in parts if hasattr(part, "text") and part.text)
#             else:
#                 print("⚠️ Gemini returned no text parts — finish_reason:",
#                 response.candidates[0].finish_reason)
#                 content = ""
#         else:
#             print("⚠️ Gemini returned no candidates at all.")
#             content = ""
      

#         code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
#         extracted_code = code_match.group(1) if code_match else content
# # Ensure numpy import exists
#         if "import numpy as np" not in extracted_code:
#             extracted_code = "import numpy as np\n" + extracted_code

# # Ensure scipy.io import for .mat files
#         if file_type.lower() == "mat" and "from scipy.io import loadmat" not in extracted_code:
#             extracted_code = "from scipy.io import loadmat\n" + extracted_code
#         if "X =" not in extracted_code:
#             extracted_code += "\n\n# Safety: use all columns except last as X"
#             extracted_code += "\ntry:\n"
#             extracted_code += "    df = pd.read_csv(file_path)\n"
#             extracted_code += "    X = df.iloc[:, :-1].values\n"
#             extracted_code += "    y = df.iloc[:, -1].values\n"
#             extracted_code += "except Exception as e:\n"
#             extracted_code += "    print(f'Fallback failed: {e}')\n"
#             extracted_code += "    X = np.empty((0,0))\n"
#             extracted_code += "    y = 'Unsupervised'\n"

#         if self.store_script:
#             with open('head_' + self.store_path, "w") as f:
#                 f.write(extracted_code)

#         return extracted_code
#     def generate_script(self):
#         """
#         Generates a Python script using GPT-4 to load a data file and extract its content.
#         """

#         # Ensure self.filepath is correctly formatted
#         file_path = self.filepath.replace("\\", "/")  # Normalize for cross-platform compatibility
#         file_type = self.filepath.split('.')[-1]  # Extract file extension

#         prompt = f"""
#             Write a complete Python script for the given dataset:
# 1. **Required imports**: os, scipy.io, pandas, json, numpy. Add torch if the extension is `.pt`.
# 2. File path: "{file_path}" (already known). Do not ask for user input.
# 3. Immediately check if the file exists with `os.path.exists("{file_path}")`.  
#    If it does not exist, print a clear error message and exit.
# 4. Load the file based on its extension `{file_type}`:
#    - `.mat`: use scipy.io.loadmat
#    - `.csv`: use pandas.read_csv
#    - `.json`: use json.load(open(file_path, 'r'))
#    - `.pt`: use torch.load(file_path, weights_only=False)
# 5. Extract **features** into a variable called `X` and **labels** into a variable called `y`:
#    - If labels are not clearly present, set `y = "Unsupervised"`.
#    - If labels are present, ensure `y` is a 1D NumPy array, and `X` is a 2D NumPy array.
#    After loading the file:
#     import pandas as pd
#     df = pd.read_csv(file_path)
#     X = df.iloc[:, :-1].values
#     y = df.iloc[:, -1].values
# 6. **MANDATORY**:  
#    - At the end of the script, `X` and `y` **must** exist in `locals()` with the exact variable names.
#    - Never leave `X` or `y` undefined.
#    - Never rely on "if X" or "if y" to check NumPy arrays — use `.shape`, `.size`, or explicit None checks.
#    - If unsure about columns, treat all columns except the target as features.
#    If they do not exist, define them explicitly:
#    - For datasets without labels, use y = "Unsupervised".
#    - For graph datasets, use y = "graph".
#    - Ensure X is a valid object (NumPy array, Pandas DataFrame values, or loaded object).
#     Never leave X or y undefined.

# 7. Do **not** include any conditional code for file type — use the given file type only.

# Return only the Python code.
# """


#         genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#         model = genai.GenerativeModel("gemini-2.5-pro")
#         response = model.generate_content(prompt)
#         # content = response.text
#         if hasattr(response, "candidates") and response.candidates:
#             parts = response.candidates[0].content.parts
#             if parts:
#                 content = "".join(part.text for part in parts if hasattr(part, "text") and part.text)
#             else:
#                 print("⚠️ Gemini returned no text parts — finish_reason:",
#                 response.candidates[0].finish_reason)
#                 content = ""
#         else:
#             print("⚠️ Gemini returned no candidates at all.")
#             content = ""

 

#         # return extracted_code
#         code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
#         extracted_code = code_match.group(1) if code_match else content
#         if "X =" not in extracted_code:
#             extracted_code += "\n\n# Safety: use all columns except last as X"
#             extracted_code += "\ntry:\n"
#             extracted_code += "    df = pd.read_csv(file_path)\n"
#             extracted_code += "    X = df.iloc[:, :-1].values\n"
#             extracted_code += "    y = df.iloc[:, -1].values\n"
#             extracted_code += "except Exception as e:\n"
#             extracted_code += "    print(f'Fallback failed: {e}')\n"
#             extracted_code += "    X = np.empty((0,0))\n"
#             extracted_code += "    y = 'Unsupervised'\n"

#         # 🔹 SAFETY: Ensure X and y are always defined
#         if "X =" not in extracted_code:
#             extracted_code += "\n\n# Safety fallback: treat all columns as features\n"
#             extracted_code += "X = df.values if 'df' in locals() else np.empty((0, 0))\n"
#         if "y =" not in extracted_code:
#             extracted_code += "\n# Safety fallback: mark as Unsupervised if y not found\n"
#             extracted_code += "y = 'Unsupervised'\n"

#         # Ensure numpy import exists
#         if "import numpy as np" not in extracted_code:
#             extracted_code = "import numpy as np\n" + extracted_code

# # Ensure scipy.io import for .mat files
#         if file_type.lower() == "mat" and "from scipy.io import loadmat" not in extracted_code:
#             extracted_code = "from scipy.io import loadmat\n" + extracted_code


#         if self.store_script:
#             with open('head_' + self.store_path, "w") as f:
#                 f.write(extracted_code)

#         return extracted_code
#     def generate_graph_script(self):
#         """
#         Generates a Python script using GPT-4 to load a data file and extract its content.
#         """

#         # Ensure self.filepath is correctly formatted
#         file_path = self.filepath.replace("\\", "/")  # Normalize for cross-platform compatibility
#         file_type = self.filepath.split('.')[-1]  # Extract file extension

#         prompt = f"""
# Write a Python script that: 
# the file is highly likely a graph data.

# 1. **Includes all necessary imports** (`os`, `scipy.io`, `pandas`, `json`, `numpy`).
# 2. Determines the file type based on the extension: `{file_type}`.
# 3. Load the file using the appropriate method. For example: 
#     `torch.load("{file_path}", weights_only=False)` for `.pt`
#     for other file, use proper way to load the data
# 4. store the data in variable call `X`, and set y = "graph"
# 5. Ensure the script runs correctly when executed like:

#         exec(generated_script, {{}}, local_namespace)
#         X = local_namespace.get("X")
#         y = local_namespace.get("y")

# Do not generate if statment code for file type because file type is already given.

# After loading the file:
#     import pandas as pd
#     df = pd.read_csv(file_path)
#     head = df.head()  # MUST assign to variable named 'head'



# **Return only the Python code.**

# """


#         # Initialize OpenAI client
#         # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         # client = genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyDzl7RXk0Gn6hlWHDiu5CkFoFfhLkv-D-c"))
#         genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#         model = genai.GenerativeModel("gemini-2.5-pro")
#         response = model.generate_content(prompt)
#         # content = response.text
#         if hasattr(response, "candidates") and response.candidates:
#             parts = response.candidates[0].content.parts
#             if parts:
#                 content = "".join(part.text for part in parts if hasattr(part, "text") and part.text)
#             else:
#                 print("⚠️ Gemini returned no text parts — finish_reason:",
#                 response.candidates[0].finish_reason)
#                 print("DEBUG messages =", type(content), content)

#                 content = ""
#         else:
#             print("⚠️ Gemini returned no candidates at all.")
#             content = ""


#         # return extracted_code
#         code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)


# #         return extracted_code
#         extracted_code = code_match.group(1) if code_match else content

#         # Ensure numpy import exists
#         if "import numpy as np" not in extracted_code:
#             extracted_code = "import numpy as np\n" + extracted_code

#         # Ensure scipy.io import for .mat files
#         if file_type.lower() == "mat" and "from scipy.io import loadmat" not in extracted_code:
#             extracted_code = "from scipy.io import loadmat\n" + extracted_code

#         # 🔹 SAFETY: Ensure X and y are always defined
#         if "X =" not in extracted_code:
#             extracted_code += "\n\n# Safety fallback: treat all columns as features\n"
#             extracted_code += "X = df.values if 'df' in locals() else np.empty((0, 0))\n"
#         if "y =" not in extracted_code:
#             extracted_code += "\n# Safety fallback: mark as graph if y not found\n"
#             extracted_code += "y = 'graph'\n"


#         # Save generated script
#         if self.store_script:
#             with open(self.store_path, "w") as f:
#                 f.write(extracted_code)

#         return extracted_code

   

#     def load_data(self, split_data=False):


#         import scipy.io
#         import torch
#         """
#     Robustly load data from CSV, MAT, NPY, or PT files.
#     Ensures X and y are always defined, even on errors.
#     Optionally splits into train/test sets.
#     """

#     # Ensure file exists
#         if not os.path.exists(self.filepath):
#             print(f"❌ File not found: {self.filepath}")
#             return np.empty((0, 0)), "Unsupervised"

#     # Determine file extension
#         file_ext = os.path.splitext(self.filepath)[1].lower()
#         X, y = None, None

#         try:
#             if file_ext == ".csv":
#             # Try UTF-8, fallback to latin1
#                 try:
#                     df = pd.read_csv(self.filepath, encoding='utf-8')
#                 except UnicodeDecodeError:
#                     df = pd.read_csv(self.filepath, encoding='latin1')
#                 X = df.iloc[:, :-1].values
#                 y = df.iloc[:, -1].values
#                 print(f"✅ Loaded CSV: {X.shape} features, {y.shape} labels")

#             elif file_ext == ".mat":
#                 mat_data = scipy.io.loadmat(self.filepath)
#                 arrays = [v for k, v in mat_data.items() if not k.startswith("__")]
#                 if len(arrays) >= 1:
#                     X = arrays[0]
#                 if len(arrays) >= 2:
#                     y = arrays[1]
#                 if y is None:
#                     y = "Unsupervised"
#                 print(f"✅ Loaded MATLAB file: X shape {X.shape if hasattr(X, 'shape') else 'unknown'}")

#             elif file_ext == ".npy":
#                 X = np.load(self.filepath, allow_pickle=True)
#                 y = "time-series"
#                 print(f"✅ Loaded Numpy array: {X.shape}")

#             # elif file_ext == ".pt":
#             #     X = torch.load(self.filepath)
#             #     y = "graph"
#             #     print(f"✅ Loaded PyTorch object: {type(X)}")
#             elif file_ext == ".pt":
#                 from torch_geometric.data import Data
#                 from torch.serialization import add_safe_globals
#                 try:
                    
#                     X = torch.load(self.filepath, map_location='cpu', weights_only=False)
#                     y = "graph"
#                     print(f"✅ Loaded PyTorch Geometric object: {type(X)} with {X.num_nodes} nodes")
#                 except Exception as e:
#                     print(f"❌ Error loading PyTorch Geometric file: {e}")
#                     X, y = np.empty((0, 0)), "graph"

#             else:
#                 print(f"❌ Unsupported file format: {file_ext}")
#                 return np.empty((0, 0)), "Unsupervised"

#         except Exception as e:
#             print(f"❌ Error loading {self.filepath}: {e}")
#             X, y = np.empty((0, 0)), "Unsupervised"

#     # Safety fallback: ensure X and y are defined
#         if X is None:
#             X = np.empty((0, 0))
#         if y is None:
#             y = "Unsupervised"

#     # Optional train/test split
#         if split_data:
#             from sklearn.model_selection import train_test_split
#             if hasattr(X, "shape") and hasattr(y, "shape") and X.shape[0] == y.shape[0]:
#                 X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42
#             )
#                 print("✅ Split data into train/test sets.")
#                 return X_train, X_test, y_train, y_test

#         return X, y




# if __name__ == "__main__":
#     import sys
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#     from config.config import Config
#     os.environ['GEMINI_API_KEY'] = Config.GEMINI_API_KEY
    
#     if os.path.exists('head_generated_data_loader.py'):
#         os.remove('head_generated_data_loader.py')
#     if os.path.exists('generated_data_loader.py'):
#         os.remove('generated_data_loader.py')

#     data_loader = DataLoader("data/MSL", store_script=True)
#     X_train, y_train = data_loader.load_data(split_data=False)

#     print(X_train)
#     print(y_train)

#     print(len(X_train))
    #Run IForest on ./data/glass_train.mat and ./data/glass_test.mat with contamination=0.1





# data_loader_optimized.py

import os
import re
import numpy as np
import pandas as pd
import torch
import json
import scipy.io
from torch_geometric.data import Data

import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class DataLoader:
    """
    Optimized DataLoader supporting .csv, .mat, .npy, .pt files.
    Generates scripts (head_*.py) and safely loads data.
    """

    def __init__(self, filepath, desc='', store_script=True, store_path='generated_data_loader.py'):
        self.filepath = filepath.replace("\\", "/")
        self.desc = desc
        self.store_script = store_script
        self.store_path = store_path

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

    def generate_script(self):
        """
        Generates a dataset loader script using Gemini AI.
        Always ensures X and y exist.
        """
        file_type = self.filepath.split('.')[-1].lower()
        prompt = f"""
Write a Python script to load this dataset:
File: {self.filepath}
Type: {file_type}
- Store features in X
- Labels in y (or y='graph' or 'Unsupervised' if not available)
- Must include: import os, numpy, pandas, scipy.io, torch
- Do not ask for input, do not detect file type
- Ensure X and y exist in locals()
Return Python code only.
"""
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)

        content = ""
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts:
                content = "".join(part.text for part in parts if hasattr(part, "text") and part.text)

        code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        extracted_code = code_match.group(1) if code_match else content

        # Safety fallback
        if "X =" not in extracted_code:
            extracted_code += "\ntry:\n    df = pd.read_csv('" + self.filepath + "')\n    X = df.iloc[:,:-1].values\n    y = df.iloc[:,-1].values\nexcept:\n    X = np.empty((0,0))\n    y = 'Unsupervised'\n"
        if "y =" not in extracted_code:
            extracted_code += "\ny = 'Unsupervised'\n"

        if self.store_script:
            with open('head_' + self.store_path, "w") as f:
                f.write(extracted_code)

        return extracted_code

    def load_data(self, split_data=False):
        """
        Load dataset safely.
        Returns X, y (or X_train, X_test, y_train, y_test if split_data=True)
        """
        X, y = None, None
        ext = os.path.splitext(self.filepath)[1].lower()

        try:
            if ext == ".csv":
                try:
                    df = pd.read_csv(self.filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(self.filepath, encoding='latin1')
                X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

            elif ext == ".mat":
                mat = scipy.io.loadmat(self.filepath)
                arrays = [v for k, v in mat.items() if not k.startswith("__")]
                X = arrays[0] if len(arrays) >= 1 else np.empty((0,0))
                y = arrays[1] if len(arrays) >= 2 else "Unsupervised"

            elif ext == ".npy":
                X = np.load(self.filepath, allow_pickle=True)
                y = "time-series"

            elif ext == ".pt":
                X = torch.load(self.filepath, map_location='cpu', weights_only=False)
                y = "graph"

            else:
                print(f"❌ Unsupported file: {self.filepath}")
                X, y = np.empty((0,0)), "Unsupervised"

        except Exception as e:
            print(f"❌ Error loading {self.filepath}: {e}")
            X, y = np.empty((0,0)), "Unsupervised"

        # Safety fallback
        if X is None: X = np.empty((0,0))
        if y is None: y = "Unsupervised"

        # Optional train/test split
        if split_data and isinstance(X, np.ndarray) and isinstance(y, np.ndarray) and X.shape[0] == y.shape[0]:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test

        return X, y


if __name__ == "__main__":
    loader = DataLoader("data/MSL.csv", store_script=True)
    X, y = loader.load_data()
    print("X shape:", X.shape)
    print("y type:", type(y))
