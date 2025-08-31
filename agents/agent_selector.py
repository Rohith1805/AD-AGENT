from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
from scipy.io import loadmat
import sys
import google.generativeai as genai
from config.config import Config
import torch
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.data_loader import DataLoader
from ad_model_selection.prompts.pygod_ms_prompt import generate_model_selection_prompt_from_pygod
from ad_model_selection.prompts.pyod_ms_prompt import generate_model_selection_prompt_from_pyod
from ad_model_selection.prompts.timeseries_ms_prompt import generate_model_selection_prompt_from_timeseries
from utils.gemini_client import query_gemini

import json

class AgentSelector:

    def __init__(self, user_input):
        self.parameters = user_input['parameters']
        self.data_path_train = user_input['dataset_train']
        self.data_path_test = user_input['dataset_test']
        self.user_input = user_input

        # 1️⃣ Load training/testing data & set package_name
        self.load_data(self.data_path_train, self.data_path_test)

        # 2️⃣ Select algorithm (and tools) using Gemini if needed
        self.set_tools()

        # 3️⃣ Generate final tools list AFTER package_name is known
        self.tools = self.generate_tools(self.user_input['algorithm'])

        print(f"Package name: {self.package_name}")
        print(f"Algorithm: {self.user_input['algorithm']}")
        print(f"Tools: {self.tools}")

        # 4️⃣ Load and split docs, build vector store
        self.documents = self.load_and_split_documents()
        self.vectorstore = self.build_vectorstore(self.documents)


    # def load_data(self, train_path, test_path):
    #   ext_train = os.path.splitext(train_path)[-1].lower() if os.path.isfile(train_path) else None

    # # --- TRAIN DATA ---
    #   if ext_train == ".csv":
    #     train_loader = DataLoader(train_path, store_script=True, store_path="train_data_loader.py")
    #     X_train, y_train = train_loader.load_data(split_data=False)

    #   elif ext_train == ".npy":
    #     X_train = np.load(train_path)
    #     y_train = None

    #   elif ext_train == ".pt":
    #     data = torch.load(train_path)
    #     if hasattr(data, "x") and hasattr(data, "y"):  # typical pygod graph
    #         X_train, y_train = data.x, data.y
    #     else:
    #         X_train, y_train = data, None

    #   elif os.path.isdir(train_path):  # time-series folder (like MSL)
    #     X_train, y_train = "tslib", "time-series"

    #   else:
    #     raise ValueError(f"Unsupported train data format: {train_path}")

    #   self.X_train, self.y_train = X_train, y_train

    # # --- TEST DATA ---
    #   if test_path and os.path.exists(test_path):
    #     ext_test = os.path.splitext(test_path)[-1].lower() if os.path.isfile(test_path) else None

    #     if ext_test == ".csv":
    #         test_loader = DataLoader(test_path, store_script=True, store_path="test_data_loader.py")
    #         X_test, y_test = test_loader.load_data(split_data=False)

    #     elif ext_test == ".npy":
    #         X_test = np.load(test_path)
    #         y_test = None

    #     elif ext_test == ".pt":
    #         data = torch.load(test_path)
    #         if hasattr(data, "x") and hasattr(data, "y"):
    #             X_test, y_test = data.x, data.y
    #         else:
    #             X_test, y_test = data, None

    #     elif os.path.isdir(test_path):
    #         X_test, y_test = "tslib", "time-series"

    #     else:
    #         raise ValueError(f"Unsupported test data format: {test_path}")

    #     self.X_test, self.y_test = X_test, y_test
    #   else:
    #     self.X_test, self.y_test = None, None

    # # --- Set package name automatically ---
    #   if type(self.X_train) is str and self.X_train == "tslib":
    #     self.package_name = "tslib"
    #   elif ext_train == ".npy":
    #     self.package_name = "tslib"
    #     if self.X_train is not None and len(self.X_train.shape) > 1:
    #         num_features = self.X_train.shape[1]
    #         self.parameters["enc_in"] = num_features
    #         self.parameters["c_out"] = num_features
    #   elif ext_train == ".pt" or (type(y_train) is str and y_train == "graph"):
    #     self.package_name = "pygod"
    #   elif type(y_train) is str and y_train == "time-series":
    #     self.package_name = "darts"
    #   else:
    #     self.package_name = "pyod"



    def load_data(self, train_path, test_path):
      ext_train = os.path.splitext(train_path)[-1].lower() if os.path.isfile(train_path) else None

    # --- TRAIN DATA ---
      if ext_train == ".csv":
        train_loader = DataLoader(train_path, store_script=True, store_path="train_data_loader.py")
        X_train, y_train = train_loader.load_data(split_data=False)

      elif ext_train == ".npy":
        X_train = np.load(train_path)
        y_train = None

      elif ext_train == ".pt":
        data = torch.load(train_path)
        if hasattr(data, "x") and hasattr(data, "y"):  # typical pygod graph
            X_train, y_train = data.x, data.y
        else:
            X_train, y_train = data, None

      elif ext_train == ".mat":   # ✅ NEW: support MATLAB files
        data = loadmat(train_path)
        # assume the dataset follows standard naming conventions
        # you may need to adjust these keys depending on dataset
        if "X" in data and "y" in data:
            X_train, y_train = data["X"], data["y"].ravel()
        else:
            # fallback: pick the first two arrays
            keys = [k for k in data.keys() if not k.startswith("__")]
            X_train = data[keys[0]]
            y_train = data[keys[1]].ravel() if len(keys) > 1 else None

      elif os.path.isdir(train_path):  # time-series folder (like MSL)
        X_train, y_train = "tslib", "time-series"

      else:
        raise ValueError(f"Unsupported train data format: {train_path}")

      self.X_train, self.y_train = X_train, y_train

    # --- TEST DATA ---
      if test_path and os.path.exists(test_path):
        ext_test = os.path.splitext(test_path)[-1].lower() if os.path.isfile(test_path) else None

        if ext_test == ".csv":
            test_loader = DataLoader(test_path, store_script=True, store_path="test_data_loader.py")
            X_test, y_test = test_loader.load_data(split_data=False)

        elif ext_test == ".npy":
            X_test = np.load(test_path)
            y_test = None

        elif ext_test == ".pt":
            data = torch.load(test_path)
            if hasattr(data, "x") and hasattr(data, "y"):
                X_test, y_test = data.x, data.y
            else:
                X_test, y_test = data, None

        elif ext_test == ".mat":   # ✅ NEW: support MATLAB files
            data = loadmat(test_path)
            if "X" in data and "y" in data:
                X_test, y_test = data["X"], data["y"].ravel()
            else:
                keys = [k for k in data.keys() if not k.startswith("__")]
                X_test = data[keys[0]]
                y_test = data[keys[1]].ravel() if len(keys) > 1 else None

        elif os.path.isdir(test_path):
            X_test, y_test = "tslib", "time-series"

        else:
            raise ValueError(f"Unsupported test data format: {test_path}")

        self.X_test, self.y_test = X_test, y_test
      else:
        self.X_test, self.y_test = None, None

    # --- Set package name automatically ---
      if type(self.X_train) is str and self.X_train == "tslib":
        self.package_name = "tslib"
      elif ext_train == ".npy":
        self.package_name = "tslib"
        if self.X_train is not None and len(self.X_train.shape) > 1:
            num_features = self.X_train.shape[1]
            self.parameters["enc_in"] = num_features
            self.parameters["c_out"] = num_features
      elif ext_train == ".pt" or (type(y_train) is str and y_train == "graph"):
        self.package_name = "pygod"
      elif type(y_train) is str and y_train == "time-series":
        self.package_name = "darts"
      else:
        self.package_name = "pyod"

    def parse_gemini_choice(self, content):
      """Safely parse Gemini JSON output and extract 'choice'."""
      try:
        print("[DEBUG] Raw content before JSON parse:", repr(content))

        data = json.loads(content)
        algorithm = data.get("choice")
        if not algorithm:
            print("[WARN] Gemini did not return 'choice'. Using default algorithm.")
            algorithm = self.user_input['algorithm'][0] if self.user_input['algorithm'] else 'IForest'
        return algorithm
      except json.JSONDecodeError:
        print("[WARN] Invalid JSON from Gemini. Using default algorithm.")
        return self.user_input['algorithm'][0] if self.user_input['algorithm'] else 'IForest'

    def set_tools(self):
      user_input = self.user_input
    # If user explicitly asked for 'all'
      if user_input['algorithm'] and user_input['algorithm'][0].lower() == "all":
        self.tools = self.generate_tools(user_input['algorithm'])
        return

      name = os.path.basename(self.data_path_train)

      if self.package_name == "pyod":
        if self.X_train is None:
            raise ValueError("X_train is None, cannot proceed with pyod")
        size = self.X_train.shape[0]
        dim = self.X_train.shape[1]
        messages = generate_model_selection_prompt_from_pyod(name, size, dim)
        # print("[DEBUG] GEMINI RAW TEXT (selector):", messages.text)
        print("[DEBUG] GEMINI RAW TEXT (selector):", [msg["content"] for msg in messages])


      elif self.package_name == "pygod":
        if self.X_train is None:
          raise ValueError("X_train is None, cannot proceed with pygod")
        num_node = self.X_train.num_nodes
        num_edge = self.X_train.num_edges
        num_feature = self.X_train.num_features
        avg_degree = num_edge / num_node
        messages = generate_model_selection_prompt_from_pygod(name, num_node, num_edge, num_feature, avg_degree)
        # print("[DEBUG] GEMINI RAW TEXT (selector):", messages.text)
        try:
          parsed = json.loads(messages.text)
          print("[DEBUG] Gemini JSON (pretty):\n", json.dumps(parsed, indent=4))
        except json.JSONDecodeError:
          print("[DEBUG] Gemini raw text (unparsed):\n", messages.text)

      else:  # Time series
        if self.X_train is None or isinstance(self.X_train, str):
            self.user_input['algorithm'] = ['Autoformer']
            return
        if len(self.X_train.shape) > 1:
            self.parameters['enc_in'] = self.X_train.shape[1]
        num_signals = len(self.X_train)
        dim = self.X_train.shape[1]
        series_type = "multivariate" if dim > 1 else "univariate"
        messages = generate_model_selection_prompt_from_timeseries(name, num_signals, dim, series_type)

    # Query Gemini
      prompt = "\n".join([msg["content"] for msg in messages])
      prompt += "\n\nIMPORTANT: Reply ONLY with a valid JSON object containing keys 'reason' and 'choice'. Do not include any text outside the JSON."
      raw_text = query_gemini(prompt)
      print("\n[DEBUG] GEMINI RAW TEXT (Selector, before JSON parse):\n", repr(raw_text), "\n")
      content = raw_text  # keep for parse_gemini_choice()       
      print("[DEBUG] Gemini raw output:", content)

    # Parse choice and update
      algorithm = self.parse_gemini_choice(content)
      self.user_input['algorithm'] = [algorithm]

    def load_and_split_documents(self,folder_path="./docs"):
      """
      load ./docs txt doc, divided into small blocks。
      """
      documents = []
      text_splitter = CharacterTextSplitter(separator="\n", chunk_size=700, chunk_overlap=150)

      for filename in os.listdir(folder_path):
         if filename.startswith(self.package_name):
               file_path = os.path.join(folder_path, filename)
               with open(file_path, "r", encoding="utf-8") as file:
                  text = file.read()
                  chunks = text_splitter.split_text(text)
                  documents.extend(chunks)

      return documents
    # def build_vectorstore(self,documents):
    #   """
    #   The segmented document blocks are converted into vectors and stored in the FAISS vector database.
    #   """
    #   embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    #   vectorstore = FAISS.from_texts(documents, embedding)
    #   return vectorstore
    def build_vectorstore(self, documents):
      """
    The segmented document blocks are converted into vectors 
    and stored in the FAISS vector database.
      """
    # ✅ Force API key authentication for embeddings (skip ADC)
      os.environ["GOOGLE_API_KEY"] = Config.GEMINI_API_KEY
      embedding = GoogleGenerativeAIEmbeddings(
      model="models/embedding-001",
      google_api_key=Config.GEMINI_API_KEY,
      # transport="grpc"
      # request_parallelism=1
      transport="rest"  # ✅ avoid async gRPC transport
    )

    # ✅ Create FAISS vectorstore from docs
      vectorstore = FAISS.from_texts(documents, embedding)
      return vectorstore

    def generate_tools(self,algorithm_input):
      """Generates the tools for the agent."""
      if algorithm_input[0].lower() == "all":
        if self.package_name == "pygod":
          return ['SCAN','GAE','Radar','ANOMALOUS','ONE','DOMINANT','DONE','AdONE','AnomalyDAE','GAAN','DMGD','OCGNN','CoLA','GUIDE','CONAD','GADNR','CARD']
        elif self.package_name == "pyod":
          return ['ECOD', 'ABOD', 'FastABOD', 'COPOD', 'MAD', 'SOS', 'QMCD', 'KDE', 'Sampling', 'GMM', 'PCA', 'KPCA', 'MCD', 'CD', 'OCSVM', 'LMDD', 'LOF', 'COF', '(Incremental) COF', 'CBLOF', 'LOCI', 'HBOS', 'kNN', 'AvgKNN', 'MedKNN', 'SOD', 'ROD', 'IForest', 'INNE', 'DIF', 'FeatureBagging', 'LSCP', 'XGBOD', 'LODA', 'SUOD', 'AutoEncoder', 'VAE', 'Beta-VAE', 'SO_GAAL', 'MO_GAAL', 'DeepSVDD', 'AnoGAN', 'ALAD', 'AE1SVM', 'DevNet', 'R-Graph', 'LUNAR']
        else:
          # return ['GlobalNaiveAggregate','GlobalNaiveDrift','GlobalNaiveSeasonal']
          return ["GlobalNaiveAggregate","GlobalNaiveDrift","GlobalNaiveSeasonal","RNNModel","BlockRNNModel","NBEATSModel","NHiTSModel","TCNModel","TransformerModel","TFTModel","DLinearModel","NLinearModel","TiDEModel","TSMixerModel","LinearRegressionModel","RandomForest","LightGBMModel","XGBModel","CatBoostModel"]
      return algorithm_input

if __name__ == "__main__":
  if os.path.exists("train_data_loader.py"):
    os.remove("train_data_loader.py")
  if os.path.exists("test_data_loader.py"):
    os.remove("test_data_loader.py")
  if os.path.exists("head_train_data_loader.py"):
    os.remove("head_train_data_loader.py")
  if os.path.exists("head_test_data_loader.py"):
    os.remove("head_test_data_loader.py")
  import sys
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  from config.config import Config
  genai.configure(api_key=Config.GEMINI_API_KEY)

  user_input = {
    "algorithm": ['TimesNet'],
    "dataset_train": "./data/MSL",
    "dataset_test": "./data/MSL",
    "parameters": {
    }
  }
  agentSelector = AgentSelector(user_input= user_input)
  print(f"Tools: {agentSelector.tools}")
  print('Parameters:', agentSelector.parameters)