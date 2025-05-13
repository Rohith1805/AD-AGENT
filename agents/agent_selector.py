from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.data_loader import DataLoader

from ad_model_selection.prompts.pygod_ms_prompt import generate_model_selection_prompt_from_pygod
from ad_model_selection.prompts.pyod_ms_prompt import generate_model_selection_prompt_from_pyod
from ad_model_selection.prompts.timeseries_ms_prompt import generate_model_selection_prompt_from_timeseries
from utils.openai_client import query_openai
import json

class AgentSelector:
    def __init__(self, user_input):
      self.parameters = user_input['parameters']
      self.data_path_train = user_input['dataset_train']
      self.data_path_test = user_input['dataset_test']
      self.user_input = user_input

      # if user_input['dataset_train'].endswith(".pt"):
      #   self.package_name = "pygod"
      # elif user_input['dataset_train'].endswith(".mat"):
      #   self.package_name = "pyod"
      # elif user_input['dataset_train'].endswith("_train.npy"):
      #   user_input['dataset_train'] = user_input['dataset_train'].replace("_train.npy", "")
      #   self.package_name = "tslib"
      # else:
      #   self.package_name = "darts"


      # self.tools = self.generate_tools(user_input['algorithm'])

      self.load_data(self.data_path_train, self.data_path_test)
      self.set_tools()

      print(f"Package name: {self.package_name}")
      print(f"Algorithm: {user_input['algorithm']}")
      print(f"Tools: {self.tools}")

      
      self.documents = self.load_and_split_documents()
      self.vectorstore = self.build_vectorstore(self.documents)

    def load_data(self, train_path, test_path):
      train_loader = DataLoader(train_path, store_script=True, store_path='train_data_loader.py')
      test_loader = DataLoader(test_path, store_script=True, store_path='test_data_loader.py')

      X_train, y_train = train_loader.load_data(split_data=False)
      X_test, y_test = test_loader.load_data(split_data=False)
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test

      if train_path.endswith('.npy'):
        self.package_name = "tslib"
      elif train_path.endswith('.pt') or type(y_train) is str and y_train == 'graph':
        self.package_name = "pygod"
      elif type(y_train) is str and y_train == 'time-series':
        self.package_name = "darts"
      else:
        self.package_name = "pyod"

    def set_tools(self):
      user_input = self.user_input
      if user_input['algorithm'] and user_input['algorithm'][0].lower() in ["all", 'none']:
        self.tools = self.generate_tools(user_input['algorithm'])
      else:
        name = os.path.basename(self.data_path_train)
        if self.package_name == "pyod":
          size = self.X_train.shape[0]
          dim = self.X_train.shape[1]
          messages = generate_model_selection_prompt_from_pyod(name, size, dim)
          content = query_openai(messages, model="o4-mini")
          algorithm = json.loads(content)["choice"]
        elif self.package_name == 'pygod':
          num_node = self.X_train.num_nodes
          num_edge = self.X_train.num_edges
          num_feature = self.X_train.num_features
          avg_degree = num_edge / num_node
          print(f"num_node: {num_node}, num_edge: {num_edge}, num_feature: {num_feature}, avg_degree: {avg_degree}")
          messages = generate_model_selection_prompt_from_pygod(name, num_node, num_edge, num_feature, avg_degree)
          content = query_openai(messages, model="o4-mini")
          algorithm = json.loads(content)["choice"]
          print(f"Algorithm: {algorithm}")
        else: # for time series data
          if self.X_train is not None:
            num_signals = len(self.X_train)
            messages = generate_model_selection_prompt_from_timeseries(name, num_signals)
            content = query_openai(messages, model="o4-mini")
            algorithm = json.loads(content)["choice"]
            print(f"Algorithm: {algorithm}")
          else:
            algorithm = 'Autoformer'

        self.tools = self.generate_tools([algorithm])
        

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
    def build_vectorstore(self,documents):
      """
      The segmented document blocks are converted into vectors and stored in the FAISS vector database.
      """
      embedding = OpenAIEmbeddings()
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
  os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY

  user_input = {
    "algorithm": [],
    "dataset_train": "./data/yahoo_train.csv",
    "dataset_test": "./data/yahoo_test.csv",
    "parameters": {
      "contamination": 0.1
    }
  }
  agentSelector = AgentSelector(user_input= user_input)
  print(f"Tools: {agentSelector.tools}")