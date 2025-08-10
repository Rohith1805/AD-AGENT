import google.generativeai as genai
import os
import re
from sklearn.model_selection import train_test_split
import pandas as pd

# import sklearn
# from orion.data import load_signal


class DataLoader:
    """
    A class to load various file formats (.mat, .csv, .json, etc.) and extract data.
    """

    def __init__(self, filepath, desc='', store_script = False, store_path = 'generated_data_loader.py'):
        """
        Initialize DataLoader with the path to the file.
        """
        self.filepath = filepath
        self.desc = desc

        self.X_name = 'X'
        self.y_name = 'y'


        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        self.store_script = store_script
        self.store_path = store_path
        self.head = None
    def generate_script_for_data_head(self):
        file_path = self.filepath.replace("\\", "/")  # Normalize for cross-platform compatibility
        file_type = self.filepath.split('.')[-1]  # Extract file extension

        prompt = f"""
Write a complete Python script for the given graph dataset:

1. **Required imports**: os, scipy.io, pandas, json, numpy. Add torch if the extension is `.pt`.
2. File path: "{file_path}" (already known). Do not ask for user input.
3. Immediately check if the file exists with `os.path.exists("{file_path}")`.  
   If it does not exist, print a clear error message and exit.
4. Load the file based on its extension `{file_type}` without using if/elif for type detection in the script. 
5. Store the data in a variable called `X`.  
   Set `y = "graph"` (always).
6. **MANDATORY**:
   - At the end of the script, `X` and `y` **must** exist in `locals()` with exactly those names.
   - Never leave `X` or `y` undefined.
   - Ensure `X` is a valid object (NumPy array or loaded graph object).
7. Do **not** include any logic to guess dataset type — it is already given.

Return only the Python code.
"""

         # Initialize OpenAI client
        # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # client = genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyDzl7RXk0Gn6hlWHDiu5CkFoFfhLkv-D-c"))
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        # Get response from GPT
        response = model.generate_content(prompt)
        # response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[
        #         {"role": "system", "content": "You are an expert Python developer."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0,
        # )
        # content = response.text
#         response = model.generate_content(prompt)

# # ✅ Safe extraction
#         if response.candidates and response.candidates[0].content.parts:
#             content = "".join(part.text for part in response.candidates[0].content.parts if part.text)
#         else:
#             print("⚠️ Gemini returned no text, using empty string.")
#             content = ""
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts:
                content = "".join(part.text for part in parts if hasattr(part, "text") and part.text)
            else:
                print("⚠️ Gemini returned no text parts — finish_reason:",
                response.candidates[0].finish_reason)
                content = ""
        else:
            print("⚠️ Gemini returned no candidates at all.")
            content = ""
        # Extract only the Python code using regex
        # code_match = re.search(r"```python\n(.*?)\n```", response.choices[0].message.content, re.DOTALL)

        # if code_match:
        #     extracted_code = code_match.group(1)
        # else:
        #     extracted_code = response.choices[0].message.content  # Fallback

        # if self.store_script:
        #     # Save the generated script for debugging
        #     with open('head_' + self.store_path, "w") as f:
        #         f.write(extracted_code)

        # # Print the generated script
        # # print("Generated Script:\n", extracted_code)

        # return extracted_code

        code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        extracted_code = code_match.group(1) if code_match else content
# Ensure numpy import exists
        if "import numpy as np" not in extracted_code:
            extracted_code = "import numpy as np\n" + extracted_code

# Ensure scipy.io import for .mat files
        if file_type.lower() == "mat" and "from scipy.io import loadmat" not in extracted_code:
            extracted_code = "from scipy.io import loadmat\n" + extracted_code
        if "X =" not in extracted_code:
            extracted_code += "\n\n# Safety: use all columns except last as X"
            extracted_code += "\ntry:\n"
            extracted_code += "    df = pd.read_csv(file_path)\n"
            extracted_code += "    X = df.iloc[:, :-1].values\n"
            extracted_code += "    y = df.iloc[:, -1].values\n"
            extracted_code += "except Exception as e:\n"
            extracted_code += "    print(f'Fallback failed: {e}')\n"
            extracted_code += "    X = np.empty((0,0))\n"
            extracted_code += "    y = 'Unsupervised'\n"

        if self.store_script:
            with open('head_' + self.store_path, "w") as f:
                f.write(extracted_code)

        return extracted_code
    def generate_script(self):
        """
        Generates a Python script using GPT-4 to load a data file and extract its content.
        """

        # Ensure self.filepath is correctly formatted
        file_path = self.filepath.replace("\\", "/")  # Normalize for cross-platform compatibility
        file_type = self.filepath.split('.')[-1]  # Extract file extension

        prompt = f"""
            Write a complete Python script for the given dataset:
1. **Required imports**: os, scipy.io, pandas, json, numpy. Add torch if the extension is `.pt`.
2. File path: "{file_path}" (already known). Do not ask for user input.
3. Immediately check if the file exists with `os.path.exists("{file_path}")`.  
   If it does not exist, print a clear error message and exit.
4. Load the file based on its extension `{file_type}`:
   - `.mat`: use scipy.io.loadmat
   - `.csv`: use pandas.read_csv
   - `.json`: use json.load(open(file_path, 'r'))
   - `.pt`: use torch.load(file_path, weights_only=False)
5. Extract **features** into a variable called `X` and **labels** into a variable called `y`:
   - If labels are not clearly present, set `y = "Unsupervised"`.
   - If labels are present, ensure `y` is a 1D NumPy array, and `X` is a 2D NumPy array.
   After loading the file:
    import pandas as pd
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
6. **MANDATORY**:  
   - At the end of the script, `X` and `y` **must** exist in `locals()` with the exact variable names.
   - Never leave `X` or `y` undefined.
   - Never rely on "if X" or "if y" to check NumPy arrays — use `.shape`, `.size`, or explicit None checks.
   - If unsure about columns, treat all columns except the target as features.
   If they do not exist, define them explicitly:
   - For datasets without labels, use y = "Unsupervised".
   - For graph datasets, use y = "graph".
   - Ensure X is a valid object (NumPy array, Pandas DataFrame values, or loaded object).
    Never leave X or y undefined.

7. Do **not** include any conditional code for file type — use the given file type only.

Return only the Python code.
"""



        # Initialize OpenAI client
        # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # client = genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyDzl7RXk0Gn6hlWHDiu5CkFoFfhLkv-D-c"))

        # Get response from GPT
        # response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[
        #         {"role": "system", "content": "You are an expert Python developer."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0,
        # )
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        # content = response.text
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts:
                content = "".join(part.text for part in parts if hasattr(part, "text") and part.text)
            else:
                print("⚠️ Gemini returned no text parts — finish_reason:",
                response.candidates[0].finish_reason)
                content = ""
        else:
            print("⚠️ Gemini returned no candidates at all.")
            content = ""

        # Extract only the Python code using regex
        # code_match = re.search(r"```python\n(.*?)\n```", response.choices[0].message.content, re.DOTALL)

        # if code_match:
        #     extracted_code = code_match.group(1)
        # else:
        #     extracted_code = response.choices[0].message.content  # Fallback

        # if self.store_script:
        #     # Save the generated script for debugging
        #     with open(self.store_path, "w") as f:
        #         f.write(extracted_code)

        # # Print the generated script
        # # print("Generated Script:\n", extracted_code)

        # return extracted_code
        code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        extracted_code = code_match.group(1) if code_match else content
        if "X =" not in extracted_code:
            extracted_code += "\n\n# Safety: use all columns except last as X"
            extracted_code += "\ntry:\n"
            extracted_code += "    df = pd.read_csv(file_path)\n"
            extracted_code += "    X = df.iloc[:, :-1].values\n"
            extracted_code += "    y = df.iloc[:, -1].values\n"
            extracted_code += "except Exception as e:\n"
            extracted_code += "    print(f'Fallback failed: {e}')\n"
            extracted_code += "    X = np.empty((0,0))\n"
            extracted_code += "    y = 'Unsupervised'\n"

        # 🔹 SAFETY: Ensure X and y are always defined
        if "X =" not in extracted_code:
            extracted_code += "\n\n# Safety fallback: treat all columns as features\n"
            extracted_code += "X = df.values if 'df' in locals() else np.empty((0, 0))\n"
        if "y =" not in extracted_code:
            extracted_code += "\n# Safety fallback: mark as Unsupervised if y not found\n"
            extracted_code += "y = 'Unsupervised'\n"

        # Ensure numpy import exists
        if "import numpy as np" not in extracted_code:
            extracted_code = "import numpy as np\n" + extracted_code

# Ensure scipy.io import for .mat files
        if file_type.lower() == "mat" and "from scipy.io import loadmat" not in extracted_code:
            extracted_code = "from scipy.io import loadmat\n" + extracted_code


        if self.store_script:
            with open('head_' + self.store_path, "w") as f:
                f.write(extracted_code)

        return extracted_code
    def generate_graph_script(self):
        """
        Generates a Python script using GPT-4 to load a data file and extract its content.
        """

        # Ensure self.filepath is correctly formatted
        file_path = self.filepath.replace("\\", "/")  # Normalize for cross-platform compatibility
        file_type = self.filepath.split('.')[-1]  # Extract file extension

        prompt = f"""
Write a Python script that: 
the file is highly likely a graph data.

1. **Includes all necessary imports** (`os`, `scipy.io`, `pandas`, `json`, `numpy`).
2. Determines the file type based on the extension: `{file_type}`.
3. Load the file using the appropriate method. For example: 
    `torch.load("{file_path}", weights_only=False)` for `.pt`
    for other file, use proper way to load the data
4. store the data in variable call `X`, and set y = "graph"
5. Ensure the script runs correctly when executed like:

        exec(generated_script, {{}}, local_namespace)
        X = local_namespace.get("X")
        y = local_namespace.get("y")

Do not generate if statment code for file type because file type is already given.

After loading the file:
    import pandas as pd
    df = pd.read_csv(file_path)
    head = df.head()  # MUST assign to variable named 'head'



**Return only the Python code.**

"""


        # Initialize OpenAI client
        # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # client = genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyDzl7RXk0Gn6hlWHDiu5CkFoFfhLkv-D-c"))
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        # content = response.text
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts:
                content = "".join(part.text for part in parts if hasattr(part, "text") and part.text)
            else:
                print("⚠️ Gemini returned no text parts — finish_reason:",
                response.candidates[0].finish_reason)
                content = ""
        else:
            print("⚠️ Gemini returned no candidates at all.")
            content = ""

        # Get response from GPT
        # response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[
        #         {"role": "system", "content": "You are an expert Python developer."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0,
        # )

        # Extract only the Python code using regex
        # code_match = re.search(r"```python\n(.*?)\n```", response.choices[0].message.content, re.DOTALL)

        # if code_match:
        #     extracted_code = code_match.group(1)
        # else:
        #     extracted_code = response.choices[0].message.content  # Fallback

        # if self.store_script:
        #     # Save the generated script for debugging
        #     with open(self.store_path, "w") as f:
        #         f.write(extracted_code)

        # # Print the generated script
        # # print("Generated Script:\n", extracted_code)

        # return extracted_code
        code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
#         extracted_code = code_match.group(1) if code_match else content
# # Ensure numpy import exists
#         if "import numpy as np" not in extracted_code:
#             extracted_code = "import numpy as np\n" + extracted_code

# # Ensure scipy.io import for .mat files
#         if file_type.lower() == "mat" and "from scipy.io import loadmat" not in extracted_code:
#             extracted_code = "from scipy.io import loadmat\n" + extracted_code

#         if self.store_script:
#             with open('head_' + self.store_path, "w") as f:
#                 f.write(extracted_code)

#         return extracted_code
        extracted_code = code_match.group(1) if code_match else content

        # Ensure numpy import exists
        if "import numpy as np" not in extracted_code:
            extracted_code = "import numpy as np\n" + extracted_code

        # Ensure scipy.io import for .mat files
        if file_type.lower() == "mat" and "from scipy.io import loadmat" not in extracted_code:
            extracted_code = "from scipy.io import loadmat\n" + extracted_code

        # 🔹 SAFETY: Ensure X and y are always defined
        if "X =" not in extracted_code:
            extracted_code += "\n\n# Safety fallback: treat all columns as features\n"
            extracted_code += "X = df.values if 'df' in locals() else np.empty((0, 0))\n"
        if "y =" not in extracted_code:
            extracted_code += "\n# Safety fallback: mark as graph if y not found\n"
            extracted_code += "y = 'graph'\n"


        # Save generated script
        if self.store_script:
            with open(self.store_path, "w") as f:
                f.write(extracted_code)

        return extracted_code

    # def load_data(self, split_data=False):
    #     """
    #     Load the data from the specified file using the generated script.
    #     The script is dynamically generated to include necessary imports and extract 'X' and 'y'.
    #     """

    #     # It is hard to load tslib data using the generated script, so we need to handle it separately.
    #     # TODO: research about better way to load the data for tslib
    #     tslib_data = ['MSL', 'PSM', 'SMAP', 'SMD', 'SWaT']
    #     if any(self.filepath.endswith(ds) for ds in tslib_data):
    #         X = 'tslib'
    #         Y = 'tslib'
    #         return X, Y


    #     if self.store_script and  'head_' + self.store_path and os.path.exists('head_' + self.store_path):
    #         head_script = open('head_' + self.store_path).read()
    #     else:
    #         head_script = self.generate_script_for_data_head()

    #     # print("Head Script:\n", head_script)
        
    #     local_namespace = {}
    #     try:
    #         # Execute the generated script safely
    #         exec(head_script, local_namespace)
            
    #         # Retrieve head from the executed script
    #         head = local_namespace.get("head")

    #         # Print the extracted data
    #         if head is not None:
    #             # print("✅ Extracted head:\n", head)
    #             self.head = head
    #         else:
    #             print("⚠️ Warning: 'head' not found in the file.")
    #     except Exception as e:
    #         print(f"❌ Error executing the generated script: {e}")
    #         return None, None

    #     # ## determine if the head is time series
    #     # if 'tiemstamp' in self.head.lower() or 'time' in self.head.lower():
    #     #     return 'time-series', 'time-series'

    #     if self.store_script and self.store_path and os.path.exists(self.store_path):
    #         generated_script = open(self.store_path).read()
    #     else:
    #         if self.head == 'graph':
    #             generated_script = self.generate_graph_script()
    #         else:
    #             generated_script = self.generate_script()


    #     # Create a controlled execution namespace
    #     local_namespace = {}

    #     try:
    #         # Execute the generated script safely
    #         exec(generated_script, local_namespace, local_namespace)
            
    #         # Retrieve X and y from the executed script
    #         X = local_namespace.get("X")
    #         y = local_namespace.get("y")


    #         # Print the extracted data
    #         # if X is not None:
    #         #     print("✅ Extracted X:\n", X)
    #         # else:
    #         #     print("⚠️ Warning: 'X' not found in the file.")
            
    #         if type(y) is str and y == 'graph':
    #             return X, y
            
    #         if type(y) is str and y == 'Unsupervised':
    #             # print("✅ Extracted y as 'Unsupervised'.")
    #             if split_data:
    #                 return X, None, None, None
    #             else:
    #                 return X, None
            
    #         if 'tiemstamp' in self.head.lower() or 'time' in self.head.lower():
    #             return X, 'time-series'

    #         # if y is not None:
    #         #     print("✅ Extracted y:\n", y)
    #         # else:
    #         #     print("⚠️ Warning: 'y' not found in the file.")

    #         # Reshape y properly
    #         if y.shape[0] == 1 and y.shape[1] == X.shape[0]:  # If y is (1, N), reshape to (N, 1)
    #             y = y.T  # Transpose to (N, 1)

    #         # Convert y to 1D if required by train_test_split
    #         if len(y.shape) > 1 and y.shape[1] == 1:
    #             y = y.ravel()


    #         # Ensure X and y now have matching samples
            
    #         if split_data:
    #             # Split the data into training and testing sets
    #             if X.shape[0] != y.shape[0]:
    #                 print(f"❌ Error: Mismatched samples. X has {X.shape[0]} rows, y has {y.shape[0]} rows.")
    #                 return None, None, None, None
    #             # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    #             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #             print("✅ Split data into training and testing sets.")
    #             return X_train, X_test, y_train, y_test
    #         else:
    #             return X, y  # Return extracted data for further processing

    #     except Exception as e:
    #         print(f"❌ Error executing the generated script: {e}")
    #         return None, None


    # def load_data(self, split_data=False):
    #     """
    #     Load the data from the specified file using the generated script.
    #     """

    #     tslib_data = ['MSL', 'PSM', 'SMAP', 'SMD', 'SWaT']
    #     if any(self.filepath.endswith(ds) for ds in tslib_data):
    #         print("[DEBUG] Detected tslib dataset.")
    #         return 'tslib', 'tslib'

    # # Step 1: Load head script
    #     if self.store_script and 'head_' + self.store_path and os.path.exists('head_' + self.store_path):
    #         head_script = open('head_' + self.store_path).read()
    #     else:
    #         head_script = self.generate_script_for_data_head()

    #     local_namespace = {}
    #     try:
    #         exec(head_script, local_namespace)
    #         self.head = local_namespace.get("head")
    #         if self.head is None:
    #             try:
                    
    #                 df = pd.read_csv(self.filepath)
    #                 X_full = df.iloc[:, :-1].values   # All columns except last → features
    #                 y_full = df.iloc[:, -1].values    # Last column → target/label

    #                 self.head = df.head()
    #                 self.X_full = X_full
    #                 self.y_full = y_full
    #                 print("[DEBUG] Fallback: Head loaded directly from CSV")
    #             except Exception as e:
    #                 print(f"[DEBUG] Head fallback failed: {e}")
    #             # ✅ Detect dataset type safely
    #         if isinstance(self.head, str) and self.head.lower() == 'graph':
    #             self.data_type = 'graph'
    #         else:
    #             self.data_type = 'table'  # or 'time-series' etc.
    #         print(f"[DEBUG] self.head type: {type(self.head)}, value: {self.head}")
    #         print(f"[DEBUG] self.data_type: {self.data_type}")
    #     except Exception as e:
    #         print(f"❌ Error executing head script: {e}")
    #         return None, None
        
    # # Step 2: Load generated script
    #     if self.store_script and self.store_path and os.path.exists(self.store_path):
    #         generated_script = open(self.store_path).read()
    #     else:
    #         if self.data_type == 'graph':
    #             generated_script = self.generate_graph_script()
    #         else:
    #             generated_script = self.generate_script()
    #     # ✅ Debug: show first few lines of the generated script
    #     print("\n[DEBUG] First 20 lines of generated script:\n")
    #     for i, line in enumerate(generated_script.splitlines()[:20], start=1):
    #         print(f"{i:02d}: {line}")
    #     print("\n")
    #     local_namespace = {}
    #     try:
    #         exec(generated_script, local_namespace, local_namespace)
    #         X = local_namespace.get("X")
    #         y = local_namespace.get("y")

    #         if X is None or y is None:
    #             try:
                    
    #                 df = pd.read_csv(self.filepath)
    #                 X = df.iloc[:, :-1].values
    #                 y = df.iloc[:, -1].values
    #                 print("[DEBUG] Fallback: X/y loaded directly from CSV")
    #             except Exception as e:
    #                 print(f"[DEBUG] X/y fallback failed: {e}")
    #                 return None, None
        #     # Step 3: Time-series detection (safe check)
        #     if isinstance(self.head, str):
        #         head_lower = self.head.lower()
        #         if 'tiemstamp' in head_lower or 'time' in head_lower:
        #             print("[DEBUG] Detected time-series dataset.")
        #             return X, 'time-series'
                
        #     if isinstance(y, str) and y == 'graph':
        #         return X, y
        #     if isinstance(y, str) and y == 'Unsupervised':
        #         return (X, None, None, None) if split_data else (X, None)

        # # Reshape y if needed
        #     if hasattr(y, 'shape'):
        #         if y.shape[0] == 1 and y.shape[1] == X.shape[0]:
        #             y = y.T
        #         if len(y.shape) > 1 and y.shape[1] == 1:
        #             y = y.ravel()

        #     if split_data:
        #         if X.shape[0] != y.shape[0]:
        #             print(f"❌ Error: Mismatched samples. X has {X.shape[0]}, y has {y.shape[0]}")
        #             return None, None, None, None
        #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #         print("✅ Split data into training/testing sets.")
        #         return X_train, X_test, y_train, y_test
        #     else:
        #         return X, y

        # except Exception as e:
        #     print(f"❌ Error executing generated script: {e}")
        #     return None, None


    def load_data(self, split_data=False):
        """
    Always load the data from local CSV (ignore Gemini-generated loader).
    """

        
        from sklearn.model_selection import train_test_split
        import os

    # Ensure file exists
        if not os.path.exists(self.filepath):
            print(f"❌ File not found: {self.filepath}")
            return None, None

        try:
        # Read full CSV
            df = pd.read_csv(self.filepath)

        # Features = all columns except last
            X = df.iloc[:, :-1].values
        # Target = last column
            y = df.iloc[:, -1].values

            print(f"[DEBUG] Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"[DEBUG] X shape: {X.shape}, y shape: {y.shape}")

        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return None, None

    # Optional split
        if split_data:
            if X.shape[0] != y.shape[0]:
                print(f"❌ Mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
                return None, None, None, None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print("✅ Split data into train/test sets.")
            return X_train, X_test, y_train, y_test

        return X, y





if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.config import Config
    os.environ['GEMINI_API_KEY'] = Config.GEMINI_API_KEY
    
    # genai.configure(api_key=Config.GEMINI_API_KEY)

    # Example usage
    # file = 'data/yahoo_train.csv'
    # from orion.data import load_signal
    # print("Loading data from:", file)
    # data = load_signal(file)
    # print(data)
    # exit()

    if os.path.exists('head_generated_data_loader.py'):
        os.remove('head_generated_data_loader.py')
    if os.path.exists('generated_data_loader.py'):
        os.remove('generated_data_loader.py')

    data_loader = DataLoader("data/MSL", store_script=True)
    X_train, y_train = data_loader.load_data(split_data=False)

    print(X_train)
    print(y_train)

    print(len(X_train))
    #Run IForest on ./data/glass_train.mat and ./data/glass_test.mat with contamination=0.1
