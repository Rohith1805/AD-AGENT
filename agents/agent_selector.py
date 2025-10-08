# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# import os
# import sys
# import google.generativeai as genai
# from config.config import Config

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from data_loader.data_loader import DataLoader
# from ad_model_selection.prompts.pygod_ms_prompt import generate_model_selection_prompt_from_pygod
# from ad_model_selection.prompts.pyod_ms_prompt import generate_model_selection_prompt_from_pyod
# from ad_model_selection.prompts.timeseries_ms_prompt import generate_model_selection_prompt_from_timeseries
# from utils.gemini_client import query_gemini

# import json

# class AgentSelector:

#     def __init__(self, user_input):
#         self.parameters = user_input['parameters']
#         self.data_path_train = user_input['dataset_train']
#         self.data_path_test = user_input['dataset_test']
#         self.user_input = user_input

#         # 1️⃣ Load training/testing data & set package_name
#         self.load_data(self.data_path_train, self.data_path_test)
 
#         # 2️⃣ Select algorithm (and tools) using Gemini if needed
#         self.set_tools()

#         # 3️⃣ Generate final tools list AFTER package_name is known
#         self.tools = self.generate_tools(self.user_input['algorithm'])

#         print(f"Package name: {self.package_name}")
#         print(f"Algorithm: {self.user_input['algorithm']}")
#         print(f"Tools: {self.tools}")

#         # 4️⃣ Load and split docs, build vector store
#         self.documents = self.load_and_split_documents()
#         self.vectorstore = self.build_vectorstore(self.documents)
#     def load_data(self, train_path, test_path):
#       train_loader = DataLoader(train_path, store_script=True, store_path='train_data_loader.py')
#       X_train, y_train = train_loader.load_data(split_data=False)
#       print(f"[DEBUG] load_data() returned: X_train={type(X_train)}, y_train={type(y_train)}")
#       self.X_train = X_train
#       self.y_train = y_train

#       # Only load test data if test_path is provided and not empty
#       if test_path and os.path.exists(test_path):
#           test_loader = DataLoader(test_path, store_script=True, store_path='test_data_loader.py')
#           X_test, y_test = test_loader.load_data(split_data=False)
#           self.X_test = X_test
#           self.y_test = y_test
#       else:
#           self.X_test = None
#           self.y_test = None

     
#       if type(self.X_train) is str and self.X_train == 'tslib':
#         self.package_name = "tslib"
#       elif train_path.endswith('.npy'):
#         self.package_name = "tslib"
#         if self.X_train is not None:
#           if len(self.X_train.shape) > 1:
#             num_features = self.X_train.shape[1]
#             self.parameters['enc_in'] = num_features
#             self.parameters['c_out'] = num_features
#       elif train_path.endswith('.pt') or type(y_train) is str and y_train == 'graph':
#         self.package_name = "pygod"
#       elif type(y_train) is str and y_train == 'time-series':
#         self.package_name = "darts"
#       else:
#         self.package_name = "pyod"

   
#     def parse_gemini_choice(self, content):
#       """Safely parse Gemini JSON output and extract 'choice'."""
#       try:
#         print("[DEBUG] Raw content before JSON parse:", repr(content))

#         data = json.loads(content)
#         algorithm = data.get("choice")
#         if not algorithm:
#             print("[WARN] Gemini did not return 'choice'. Using default algorithm.")
#             algorithm = self.user_input['algorithm'][0] if self.user_input['algorithm'] else 'IForest'
#         return algorithm
#       except json.JSONDecodeError:
#         print("[WARN] Invalid JSON from Gemini. Using default algorithm.")
#         return self.user_input['algorithm'][0] if self.user_input['algorithm'] else 'IForest'

#     def set_tools(self):
#       user_input = self.user_input
#     # If user explicitly asked for 'all'
#       if user_input['algorithm'] and user_input['algorithm'][0].lower() == "all":
#         self.tools = self.generate_tools(user_input['algorithm'])
#         return

#       name = os.path.basename(self.data_path_train)

#       if self.package_name == "pyod":
#         if self.X_train is None:
#             raise ValueError("X_train is None, cannot proceed with pyod")
#         size = self.X_train.shape[0]
#         dim = self.X_train.shape[1]
#         messages = generate_model_selection_prompt_from_pyod(name, size, dim)
#         # print("[DEBUG] GEMINI RAW TEXT (selector):", messages.text)
#         print("[DEBUG] GEMINI RAW TEXT (selector):", [msg["content"] for msg in messages])


#       elif self.package_name == "pygod":
#         if self.X_train is None:
#           raise ValueError("X_train is None, cannot proceed with pygod")
#         num_node = self.X_train.num_nodes
#         num_edge = self.X_train.num_edges
#         num_feature = self.X_train.num_features
#         avg_degree = num_edge / num_node
#         messages = generate_model_selection_prompt_from_pygod(name, num_node, num_edge, num_feature, avg_degree)
#         # print("[DEBUG] GEMINI RAW TEXT (selector):", messages.text)
#         try:
#           # parsed = json.loads(messages.text)
#           # messages comes from Gemini API response
#             if hasattr(messages, "candidates") and messages.candidates:
#               parts = messages.candidates[0].content.parts
#               if parts:
#                 content = "".join(part.text for part in parts if hasattr(part, "text") and part.text)
#                 try:
#                   parsed = json.loads(content)
#                 except json.JSONDecodeError:
#                   parsed = {}
#               else:
#                 parsed = {}
#             else:
#               parsed = {}

#             print("[DEBUG] Gemini JSON (pretty):\n", json.dumps(parsed, indent=4))
#         except json.JSONDecodeError:
#           print("[DEBUG] Gemini raw text (unparsed):\n", messages.text)

#       else:  # Time series
#         if self.X_train is None or isinstance(self.X_train, str):
#             self.user_input['algorithm'] = ['Autoformer']
#             return
#         if len(self.X_train.shape) > 1:
#             self.parameters['enc_in'] = self.X_train.shape[1]
#         num_signals = len(self.X_train)
#         dim = self.X_train.shape[1]
#         series_type = "multivariate" if dim > 1 else "univariate"
#         messages = generate_model_selection_prompt_from_timeseries(name, num_signals, dim, series_type)

#     # Query Gemini
#       prompt = "\n".join([msg["content"] for msg in messages])
#       prompt += "\n\nIMPORTANT: Reply ONLY with a valid JSON object containing keys 'reason' and 'choice'. Do not include any text outside the JSON."
#       raw_text = query_gemini(prompt)
#       print("\n[DEBUG] GEMINI RAW TEXT (Selector, before JSON parse):\n", repr(raw_text), "\n")
#       content = raw_text  # keep for parse_gemini_choice()       
#       print("[DEBUG] Gemini raw output:", content)

#     # Parse choice and update
#       algorithm = self.parse_gemini_choice(content)
#       self.user_input['algorithm'] = [algorithm]

#     def load_and_split_documents(self,folder_path="./docs"):
#       """
#       load ./docs txt doc, divided into small blocks。
#       """
#       documents = []
#       text_splitter = CharacterTextSplitter(separator="\n", chunk_size=700, chunk_overlap=150)

#       for filename in os.listdir(folder_path):
#          if filename.startswith(self.package_name):
#                file_path = os.path.join(folder_path, filename)
#                with open(file_path, "r", encoding="utf-8") as file:
#                   text = file.read()
#                   chunks = text_splitter.split_text(text)
#                   documents.extend(chunks)

#       return documents
#     # def build_vectorstore(self,documents):
#     #   """
#     #   The segmented document blocks are converted into vectors and stored in the FAISS vector database.
#     #   """
#     #   embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     #   vectorstore = FAISS.from_texts(documents, embedding)
#     #   return vectorstore
#     def build_vectorstore(self, documents):
#       """
#     The segmented document blocks are converted into vectors 
#     and stored in the FAISS vector database.
#       """
#     # ✅ Force API key authentication for embeddings (skip ADC)
#       os.environ["GOOGLE_API_KEY"] = Config.GEMINI_API_KEY
#       embedding = HuggingFaceEmbeddings(
#       model="sentence-transformers/all-MiniLM-L6-v2",
#       # google_api_key=Config.GEMINI_API_KEY,
#       # # transport="grpc"
#       # # request_parallelism=1
#       # transport="rest"  # ✅ avoid async gRPC transport
#     )

#     # ✅ Create FAISS vectorstore from docs
#       vectorstore = FAISS.from_texts(documents, embedding)
#       return vectorstore

#     def generate_tools(self,algorithm_input):
#       """Generates the tools for the agent."""
#       if algorithm_input[0].lower() == "all":
#         if self.package_name == "pygod":
#           return ['SCAN','GAE','Radar','ANOMALOUS','ONE','DOMINANT','DONE','AdONE','AnomalyDAE','GAAN','DMGD','OCGNN','CoLA','GUIDE','CONAD','GADNR','CARD']
#         elif self.package_name == "pyod":
#           return ['ECOD', 'ABOD', 'FastABOD', 'COPOD', 'MAD', 'SOS', 'QMCD', 'KDE', 'Sampling', 'GMM', 'PCA', 'KPCA', 'MCD', 'CD', 'OCSVM', 'LMDD', 'LOF', 'COF', '(Incremental) COF', 'CBLOF', 'LOCI', 'HBOS', 'kNN', 'AvgKNN', 'MedKNN', 'SOD', 'ROD', 'IForest', 'INNE', 'DIF', 'FeatureBagging', 'LSCP', 'XGBOD', 'LODA', 'SUOD', 'AutoEncoder', 'VAE', 'Beta-VAE', 'SO_GAAL', 'MO_GAAL', 'DeepSVDD', 'AnoGAN', 'ALAD', 'AE1SVM', 'DevNet', 'R-Graph', 'LUNAR']
#         else:
#           # return ['GlobalNaiveAggregate','GlobalNaiveDrift','GlobalNaiveSeasonal']
#           return ["GlobalNaiveAggregate","GlobalNaiveDrift","GlobalNaiveSeasonal","RNNModel","BlockRNNModel","NBEATSModel","NHiTSModel","TCNModel","TransformerModel","TFTModel","DLinearModel","NLinearModel","TiDEModel","TSMixerModel","LinearRegressionModel","RandomForest","LightGBMModel","XGBModel","CatBoostModel"]
#       return algorithm_input

# if __name__ == "__main__":
#   if os.path.exists("train_data_loader.py"):
#     os.remove("train_data_loader.py")
#   if os.path.exists("test_data_loader.py"):
#     os.remove("test_data_loader.py")
#   if os.path.exists("head_train_data_loader.py"):
#     os.remove("head_train_data_loader.py")
#   if os.path.exists("head_test_data_loader.py"):
#     os.remove("head_test_data_loader.py")
#   import sys
#   sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#   from config.config import Config
#   genai.configure(api_key=Config.GEMINI_API_KEY)

#   user_input = {
#     "algorithm": ['TimesNet'],
#     "dataset_train": "./data/MSL",
#     "dataset_test": "./data/MSL",
#     "parameters": {
#     }
#   }
#   agentSelector = AgentSelector(user_input= user_input)
#   print(f"Tools: {agentSelector.tools}")
#   print('Parameters:', agentSelector.parameters)
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# import os
# from scipy.io import loadmat
# import sys
# import google.generativeai as genai
# from config.config import Config
# import torch
# import numpy as np
# import pandas as pd
# import json
# import logging
# import sympy
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from data_loader.data_loader import DataLoader
# from ad_model_selection.prompts.pygod_ms_prompt import generate_model_selection_prompt_from_pygod
# from ad_model_selection.prompts.pyod_ms_prompt import generate_model_selection_prompt_from_pyod
# from ad_model_selection.prompts.timeseries_ms_prompt import generate_model_selection_prompt_from_timeseries
# from utils.gemini_client import query_gemini

# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# class AgentSelector:
#     def __init__(self, user_input):
#         self.parameters = user_input['parameters']
#         self.data_path_train = user_input['dataset_train']
#         self.data_path_test = user_input['dataset_test']
#         self.user_input = user_input

#         # Step 1️⃣ Load data
#         self.load_data(self.data_path_train, self.data_path_test)

#         # Step 2️⃣ Choose algorithm if needed
#         self.set_tools()

#         # Step 3️⃣ Generate tools list
#         self.tools = self.generate_tools(self.user_input['algorithm'])

#         print(f"Package name: {self.package_name}")
#         print(f"Algorithm: {self.user_input['algorithm']}")
#         print(f"Tools: {self.tools}")

#         # Step 4️⃣ Build vector store
#         self.documents = self.load_and_split_documents()
#         self.vectorstore = self.build_vectorstore(self.documents)

#     # -------------------------------
#     # Utility: Safe shape extraction
#     # -------------------------------
#     def _get_shape_info(self, arr):
#         if arr is None:
#             return None, None
#         try:
#             shape = arr.shape
#             if len(shape) == 1:
#                 return shape[0], 1
#             return shape[0], shape[1]
#         except Exception:
#             try:
#                 n = len(arr)
#                 return n, 1
#             except Exception:
#                 return None, None

#     # -------------------------------
#     # Utility: Normalize Gemini prompt
#     # -------------------------------
#     def _messages_to_prompt(self, messages):
#         if messages is None:
#             return ""
#         if isinstance(messages, str):
#             return messages
#         if hasattr(messages, "text"):
#             return messages.text
#         if isinstance(messages, (list, tuple)):
#             try:
#                 return "\n".join(m.get("content", str(m)) for m in messages)
#             except Exception:
#                 return "\n".join(str(m) for m in messages)
#         return str(messages)

#     # -------------------------------
#     # Data Loading
#     # -------------------------------
#     def load_data(self, train_path, test_path):
#         ext_train = os.path.splitext(train_path)[-1].lower() if os.path.isfile(train_path) else None

#         # --- TRAIN ---
#         if ext_train == ".csv":
#             train_loader = DataLoader(train_path, store_script=True, store_path="train_data_loader.py")
#             X_train, y_train = train_loader.load_data(split_data=False)

#         elif ext_train == ".npy":
#             X_train = np.load(train_path)
#             y_train = None

#         elif ext_train == ".pt":
#             # data = torch.load(train_path)
#             data = torch.load(train_path, weights_only=False)
#             # Preserve graph objects for pygod
#             if hasattr(data, "num_nodes") or hasattr(data, "edge_index"):
#                 X_train, y_train = data, getattr(data, "y", None)
#             elif hasattr(data, "x") and hasattr(data, "y"):
#                 X_train, y_train = data, data.y
#             else:
#                 X_train, y_train = data, None

#         elif ext_train == ".mat":
#             data = loadmat(train_path)
#             keys = [k for k in data.keys() if not k.startswith("__")]
#             if "X" in data and "y" in data:
#                 X_train, y_train = data["X"], data["y"].ravel()
#             elif len(keys) >= 2:
#                 X_train, y_train = data[keys[0]], data[keys[1]].ravel()
#             else:
#                 X_train, y_train = data[keys[0]], None

#         elif os.path.isdir(train_path):  # time-series folder
#             X_train, y_train = "tslib", "time-series"

#         else:
#             raise ValueError(f"Unsupported train data format: {train_path}")

#         self.X_train, self.y_train = X_train, y_train

#         # --- TEST ---
#         if test_path and os.path.exists(test_path):
#             ext_test = os.path.splitext(test_path)[-1].lower() if os.path.isfile(test_path) else None
#             if ext_test == ".csv":
#                 test_loader = DataLoader(test_path, store_script=True, store_path="test_data_loader.py")
#                 X_test, y_test = test_loader.load_data(split_data=False)
#             elif ext_test == ".npy":
#                 X_test, y_test = np.load(test_path), None
#             elif ext_test == ".pt":
#                 data = torch.load(test_path)
#                 if hasattr(data, "num_nodes") or hasattr(data, "edge_index"):
#                     X_test, y_test = data, getattr(data, "y", None)
#                 else:
#                     X_test, y_test = data, None
#             elif ext_test == ".mat":
#                 data = loadmat(test_path)
#                 keys = [k for k in data.keys() if not k.startswith("__")]
#                 if "X" in data and "y" in data:
#                     X_test, y_test = data["X"], data["y"].ravel()
#                 elif len(keys) >= 2:
#                     X_test, y_test = data[keys[0]], data[keys[1]].ravel()
#                 else:
#                     X_test, y_test = data[keys[0]], None
#             elif os.path.isdir(test_path):
#                 X_test, y_test = "tslib", "time-series"
#             else:
#                 raise ValueError(f"Unsupported test data format: {test_path}")
#             self.X_test, self.y_test = X_test, y_test
#         else:
#             self.X_test, self.y_test = None, None

#         # --- PACKAGE NAME ---
#         if isinstance(self.X_train, str) and self.X_train == "tslib":
#             self.package_name = "tslib"
#         elif ext_train == ".npy":
#             self.package_name = "tslib"
#             if hasattr(self.X_train, "shape") and len(self.X_train.shape) > 1:
#                 num_features = self.X_train.shape[1]
#                 self.parameters["enc_in"] = num_features
#                 self.parameters["c_out"] = num_features
#         elif ext_train == ".pt" and hasattr(self.X_train, "num_nodes"):
#             self.package_name = "pygod"
#         elif isinstance(self.y_train, str) and self.y_train == "time-series":
#             self.package_name = "darts"
#         else:
#             self.package_name = "pyod"

#     # -------------------------------
#     # Parse Gemini JSON
#     # -------------------------------
#     def parse_gemini_choice(self, content):
#         try:
#             logging.info(f"Gemini raw JSON candidate: {repr(content)}")
#             data = json.loads(content)
#             algorithm = data.get("choice")
#             if not algorithm:
#                 logging.warning("Gemini did not return 'choice'. Using default algorithm.")
#                 algorithm = self.user_input['algorithm'][0] if self.user_input['algorithm'] else 'IForest'
#             return algorithm
#         except json.JSONDecodeError:
#             logging.warning("Invalid JSON from Gemini. Using default algorithm.")
#             return self.user_input['algorithm'][0] if self.user_input['algorithm'] else 'IForest'

#     # -------------------------------
#     # Tool selection (Gemini)
#     # -------------------------------
#     def set_tools(self):
#         user_input = self.user_input

#         # If user explicitly requested 'all'
#         if user_input['algorithm'] and str(user_input['algorithm'][0]).lower() == "all":
#             self.tools = self.generate_tools(user_input['algorithm'])
#             return

#         name = os.path.basename(self.data_path_train)

#         if self.package_name == "pyod":
#             if self.X_train is None:
#                 raise ValueError("X_train is None, cannot proceed with pyod")
#             size, dim = self._get_shape_info(self.X_train)
#             messages = generate_model_selection_prompt_from_pyod(name, size, dim)

#         elif self.package_name == "pygod":
#             graph = self.X_train
#             if graph is None or not hasattr(graph, "num_nodes"):
#                 raise ValueError("Invalid graph object for pygod")
#             num_node = graph.num_nodes
#             num_edge = graph.num_edges
#             num_feature = graph.num_features
#             avg_degree = num_edge / num_node if num_node else 0
#             messages = generate_model_selection_prompt_from_pygod(name, num_node, num_edge, num_feature, avg_degree)

#         else:  # Time series
#             if self.X_train is None or isinstance(self.X_train, str):
#                 self.user_input['algorithm'] = ['Autoformer']
#                 return
#             size, dim = self._get_shape_info(self.X_train)
#             series_type = "multivariate" if dim > 1 else "univariate"
#             messages = generate_model_selection_prompt_from_timeseries(name, size, dim, series_type)

#         # Query Gemini
#         prompt = self._messages_to_prompt(messages)
#         prompt += "\n\nIMPORTANT: Reply ONLY with a valid JSON object containing keys 'reason' and 'choice'."
#         raw_text = query_gemini(prompt)
#         logging.info(f"Gemini raw output: {repr(raw_text)}")

#         algorithm = self.parse_gemini_choice(raw_text)
#         self.user_input['algorithm'] = [algorithm]

#     # -------------------------------
#     # Docs & Vectorstore
#     # -------------------------------
#     def load_and_split_documents(self, folder_path="./docs"):
#         documents = []
#         if not os.path.isdir(folder_path):
#             logging.warning(f"Docs folder not found: {folder_path}")
#             return documents

#         text_splitter = CharacterTextSplitter(separator="\n", chunk_size=700, chunk_overlap=150)
#         for filename in os.listdir(folder_path):
#             if filename.startswith(self.package_name):
#                 file_path = os.path.join(folder_path, filename)
#                 with open(file_path, "r", encoding="utf-8") as file:
#                     text = file.read()
#                     documents.extend(text_splitter.split_text(text))
#         return documents

#     def build_vectorstore(self, documents):
#         if not documents:
#             logging.warning("No documents found. Skipping FAISS build.")
#             return None

#         os.environ["GOOGLE_API_KEY"] = Config.GEMINI_API_KEY
#         embedding = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=Config.GEMINI_API_KEY,
#             transport="rest"
#         )
#         vectorstore = FAISS.from_texts(documents, embedding)
#         return vectorstore

#     # -------------------------------
#     # Tool generation
#     # -------------------------------
#     def generate_tools(self, algorithm_input):
#         if isinstance(algorithm_input, str):
#             alg0 = algorithm_input.lower()
#         elif isinstance(algorithm_input, (list, tuple)) and algorithm_input:
#             alg0 = str(algorithm_input[0]).lower()
#         else:
#             alg0 = ""

#         if alg0 == "all":
#             if self.package_name == "pygod":
#                 return ['SCAN','GAE','Radar','ANOMALOUS','ONE','DOMINANT','DONE','AdONE','AnomalyDAE','GAAN','DMGD','OCGNN','CoLA','GUIDE','CONAD','GADNR','CARD']
#             elif self.package_name == "pyod":
#                 return ['ECOD', 'ABOD', 'FastABOD', 'COPOD', 'MAD', 'SOS', 'QMCD', 'KDE', 'Sampling', 'GMM', 'PCA', 'KPCA', 'MCD', 'CD', 'OCSVM', 'LMDD', 'LOF', 'COF', '(Incremental) COF', 'CBLOF', 'LOCI', 'HBOS', 'kNN', 'AvgKNN', 'MedKNN', 'SOD', 'ROD', 'IForest', 'INNE', 'DIF', 'FeatureBagging', 'LSCP', 'XGBOD', 'LODA', 'SUOD', 'AutoEncoder', 'VAE', 'Beta-VAE', 'SO_GAAL', 'MO_GAAL', 'DeepSVDD', 'AnoGAN', 'ALAD', 'AE1SVM', 'DevNet', 'R-Graph', 'LUNAR']
#             else:
#                 return ["GlobalNaiveAggregate","GlobalNaiveDrift","GlobalNaiveSeasonal","RNNModel","BlockRNNModel","NBEATSModel","NHiTSModel","TCNModel","TransformerModel","TFTModel","DLinearModel","NLinearModel","TiDEModel","TSMixerModel","LinearRegressionModel","RandomForest","LightGBMModel","XGBModel","CatBoostModel"]
#         return algorithm_input


# # -------------------------------
# # MAIN EXECUTION
# # -------------------------------
# if __name__ == "__main__":
#     for f in ["train_data_loader.py", "test_data_loader.py", "head_train_data_loader.py", "head_test_data_loader.py"]:
#         if os.path.exists(f):
#             os.remove(f)

#     genai.configure(api_key=Config.GEMINI_API_KEY)
#     user_input = {
#         "algorithm": ['TimesNet'],
#         "dataset_train": "./data/MSL",
#         "dataset_test": "./data/MSL",
#         "parameters": {}
#     }

#     agentSelector = AgentSelector(user_input=user_input)
#     print(f"Tools: {agentSelector.tools}")
#     print('Parameters:', agentSelector.parameters)








# agent_selector_optimized.py

import os
import sys
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
from data_loader.data_loader import DataLoader
from ad_model_selection.prompts.pygod_ms_prompt import generate_model_selection_prompt_from_pygod
from ad_model_selection.prompts.pyod_ms_prompt import generate_model_selection_prompt_from_pyod
from ad_model_selection.prompts.timeseries_ms_prompt import generate_model_selection_prompt_from_timeseries
from utils.gemini_client import query_gemini

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class AgentSelector:

    def __init__(self, user_input):
        self.user_input = user_input
        self.data_path_train = user_input['dataset_train']
        self.data_path_test = user_input['dataset_test']
        self.parameters = user_input.get('parameters', {})

        # Load data
        self.load_data(user_input['dataset_train'], user_input.get('dataset_test'))

        # Detect package based on loaded data
        self.detect_package()

        # Set tools/algorithm using Gemini
        self.set_tools()

        # Generate final tools list
        self.tools = self.generate_tools(self.user_input['algorithm'])

        # Load and build FAISS vectorstore
        self.documents = self.load_and_split_documents()
        self.vectorstore = self.build_vectorstore(self.documents)

        # Final output
        print(f"Package: {self.package_name}")
        print(f"Algorithm: {self.user_input['algorithm']}")
        print(f"Tools: {self.tools}")

    def load_data(self, train_path, test_path=None):
        # Load training data
        train_loader = DataLoader(train_path, store_script=True, store_path='train_data_loader.py')
        self.X_train, self.y_train = train_loader.load_data(split_data=False)

        # Load test data if exists
        if test_path and os.path.exists(test_path):
            test_loader = DataLoader(test_path, store_script=True, store_path='test_data_loader.py')
            self.X_test, self.y_test = test_loader.load_data(split_data=False)
        else:
            self.X_test, self.y_test = None, None

    def detect_package(self):
        file_ext = os.path.splitext(self.data_path_train)[1].lower()
        if file_ext == ".mat":
            self.package_name = "pyod"
            return
        # Automatic detection based on file type or y
        if isinstance(self.y_train, str):
            if self.y_train == "graph":
                self.package_name = "pygod"
            elif self.y_train == "time-series":
                self.package_name = "darts"
            else:
                self.package_name = "pyod"
        elif hasattr(self.X_train, "num_nodes"):  # torch_geometric Data
            self.package_name = "pygod"
        elif isinstance(self.X_train, np.ndarray):
            self.package_name = "darts"  # assume time-series
        else:
            self.package_name = "pyod"

        # Set parameters for time-series
        if self.package_name in ["darts", "tslib"] and hasattr(self.X_train, "shape"):
            if len(self.X_train.shape) > 1:
                self.parameters['enc_in'] = self.X_train.shape[1]
                self.parameters['c_out'] = self.X_train.shape[1]

    def parse_gemini_choice(self, content):
        try:
            data = json.loads(content)
            return data.get("choice", self.user_input['algorithm'][0])
        except json.JSONDecodeError:
            return self.user_input['algorithm'][0]

    def set_tools(self):
        # If algorithm = 'all', no need for Gemini
        if self.user_input['algorithm'][0].lower() == 'all':
            return

        name = os.path.basename(self.user_input['dataset_train'])

        if self.package_name == "pyod":
            size, dim = self.X_train.shape
            messages = generate_model_selection_prompt_from_pyod(name, size, dim)
        elif self.package_name == "pygod":
            num_node = self.X_train.num_nodes
            num_edge = self.X_train.num_edges
            num_feature = self.X_train.num_features
            avg_degree = num_edge / num_node
            messages = generate_model_selection_prompt_from_pygod(name, num_node, num_edge, num_feature, avg_degree)
        else:  # time-series
            num_signals = len(self.X_train)
            dim = self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1
            series_type = "multivariate" if dim > 1 else "univariate"
            messages = generate_model_selection_prompt_from_timeseries(name, num_signals, dim, series_type)

        # Combine messages and query Gemini
        prompt = "\n".join([msg["content"] for msg in messages])
        prompt += "\n\nIMPORTANT: Reply ONLY with a valid JSON object containing keys 'reason' and 'choice'."
        raw_text = query_gemini(prompt)
        choice = self.parse_gemini_choice(raw_text)
        self.user_input['algorithm'] = [choice]

    def generate_tools(self, algorithm_input):
        if algorithm_input[0].lower() == "all":
            if self.package_name == "pygod":
                return ['SCAN','GAE','Radar','ANOMALOUS','ONE','DOMINANT','DONE','AdONE','AnomalyDAE','GAAN','DMGD','OCGNN','CoLA','GUIDE','CONAD','GADNR','CARD']
            elif self.package_name == "pyod":
                return ['ECOD', 'ABOD', 'FastABOD', 'COPOD', 'MAD', 'SOS', 'QMCD', 'KDE', 'Sampling', 'GMM', 'PCA', 'KPCA', 'MCD', 'CD', 'OCSVM', 'LMDD', 'LOF', 'COF', '(Incremental) COF', 'CBLOF', 'LOCI', 'HBOS', 'kNN', 'AvgKNN', 'MedKNN', 'SOD', 'ROD', 'IForest', 'INNE', 'DIF', 'FeatureBagging', 'LSCP', 'XGBOD', 'LODA', 'SUOD', 'AutoEncoder', 'VAE', 'Beta-VAE', 'SO_GAAL', 'MO_GAAL', 'DeepSVDD', 'AnoGAN', 'ALAD', 'AE1SVM', 'DevNet', 'R-Graph', 'LUNAR']
            else:
                return ["GlobalNaiveAggregate","GlobalNaiveDrift","GlobalNaiveSeasonal","RNNModel","BlockRNNModel","NBEATSModel","NHiTSModel","TCNModel","TransformerModel","TFTModel","DLinearModel","NLinearModel","TiDEModel","TSMixerModel","LinearRegressionModel","RandomForest","LightGBMModel","XGBModel","CatBoostModel"]
        return algorithm_input

    def load_and_split_documents(self, folder_path="./docs"):
        documents = []
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=700, chunk_overlap=150)
        for filename in os.listdir(folder_path):
            if filename.startswith(self.package_name):
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                    chunks = text_splitter.split_text(text)
                    documents.extend(chunks)
        return documents

    def build_vectorstore(self, documents):
        os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"
        embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(documents, embedding)
        return vectorstore

if __name__ == "__main__":
    user_input = {
        "algorithm": ['TimesNet'],
        "dataset_train": "./data/MSL.csv",
        "dataset_test": "./data/MSL.csv",
        "parameters": {}
    }
    agent = AgentSelector(user_input)
    print("Tools:", agent.tools)
    print("Parameters:", agent.parameters)
