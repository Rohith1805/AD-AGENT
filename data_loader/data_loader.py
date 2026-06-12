
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
