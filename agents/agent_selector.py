
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
