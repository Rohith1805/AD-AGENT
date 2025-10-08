# # import torch
# # # Load the file
# # data = torch.load("MSL_train.npy", map_location='cpu',weights_only=False)

# # print(type(data))
# # print(data)
# import torch

# # Load the file
# data = torch.load("MSL_train.npy", map_location='cpu')

# # Check the type of the object
# print(type(data))

# # If it’s a dict (common), see its keys
# if isinstance(data, dict):
#     print(data.keys())

# # Inspect a sample
# if isinstance(data, dict):
#     for k, v in data.items():
#         print(f"Key: {k}, Type: {type(v)}, Shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")





##.npy file
# import numpy as np
# import matplotlib.pyplot as plt

# # ==============================
# # 1️⃣ Load the .npy file
# # ==============================
# file_path = "MSL_train.npy"  # <-- replace with your file path
# data = np.load(file_path)

# # ==============================
# # 2️⃣ Inspect basic info
# # ==============================
# print("✅ Type of data:", type(data))
# print("✅ Shape of data:", data.shape)

# # ==============================
# # 3️⃣ Peek at first few samples
# # ==============================
# print("\n✅ First 5 samples:\n", data[:5])

# # ==============================
# # 4️⃣ Check basic statistics
# # ==============================
# print("\n✅ Data statistics:")
# print("Min:", np.min(data))
# print("Max:", np.max(data))
# print("Mean:", np.mean(data))
# print("Std:", np.std(data))

# # ==============================
# # 5️⃣ Visualize first time series sample
# # ==============================
# # Handle 1D or 2D series
# if data.ndim == 1:
#     plt.plot(data)
#     plt.title("Time Series")
#     plt.xlabel("Time")
#     plt.ylabel("Value")
# elif data.ndim == 2:
#     # Assuming shape = (timesteps, features)
#     plt.figure(figsize=(10, 5))
#     for i in range(data.shape[1]):
#         plt.plot(data[:, i], label=f"Feature {i}")
#     plt.title("First Time Series Sample")
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     plt.legend()
# elif data.ndim == 3:
#     # Assuming shape = (samples, timesteps, features)
#     plt.figure(figsize=(10, 5))
#     for i in range(data.shape[2]):
#         plt.plot(data[0, :, i], label=f"Feature {i}")
#     plt.title("First Sample from Multivariate Time Series")
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     plt.legend()
# else:
#     print("⚠️ Data has unexpected number of dimensions:", data.ndim)

# plt.show()


##.pt file 



# import torch
# from torch_geometric.data import Data
# import matplotlib.pyplot as plt

# # Load the .pt file
# file_path = "inj_cora_train.pt"
# data = torch.load(file_path, map_location='cpu', weights_only=False)

# print("✅ Type of data:", type(data))

# # List all attributes
# print("✅ Available attributes in Data object:", data.keys)  # or use data.__dict__.keys()

# # Example: inspect node features
# if hasattr(data, 'x') and data.x is not None:
#     print("\n--- Node features (x) ---")
#     print("Shape:", data.x.shape)
#     print("First 5 nodes:\n", data.x[:5])
#     print("Min:", data.x.min().item())
#     print("Max:", data.x.max().item())
#     print("Mean:", data.x.mean().item())
#     print("Std:", data.x.std().item())

# # Example: inspect labels if present
# if hasattr(data, 'y') and data.y is not None:
#     print("\n--- Labels (y) ---")
#     print("Shape:", data.y.shape)
#     print("First 5 labels:\n", data.y[:5])

# # Example: inspect edges
# if hasattr(data, 'edge_index') and data.edge_index is not None:
#     print("\n--- Edge index ---")
#     print("Shape:", data.edge_index.shape)
#     print("First 10 edges:\n", data.edge_index[:, :10])

# # Plotting first feature of nodes if it makes sense
# if hasattr(data, 'x') and data.x is not None:
#     plt.figure(figsize=(10, 5))
#     plt.plot(data.x[:, 0].numpy())  # plot first feature across nodes
#     plt.title("First Feature of Node Features")
#     plt.xlabel("Node index")
#     plt.ylabel("Feature value")
#     plt.show()


# #======================.csv files
# import pandas as pd
# import matplotlib.pyplot as plt

# # ==============================
# # 1️⃣ Load CSV file
# # ==============================
# file_path = "yahoo_sub_5.csv"  # <-- replace with your CSV file path
# df = pd.read_csv(file_path)

# # ==============================
# # 2️⃣ Inspect basic info
# # ==============================
# print("✅ Type of data:", type(df))
# print("✅ Shape of data:", df.shape)
# print("✅ Columns:", df.columns.tolist())

# # ==============================
# # 3️⃣ Peek at first few rows
# # ==============================
# print("\n✅ First 5 rows:\n", df.head())

# # ==============================
# # 4️⃣ Basic statistics
# # ==============================
# print("\n✅ Data statistics:")
# print(df.describe())

# # ==============================
# # 5️⃣ Plot selected columns (time-series)
# # ==============================
# # If you have multiple features, you can plot all or a subset
# plt.figure(figsize=(10, 5))

# # Example: plot first 3 numeric columns
# numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
# for col in numeric_cols[:3]:  # adjust number of columns to plot
#     plt.plot(df[col], label=col)

# plt.title("Time-Series Data from CSV")
# plt.xlabel("Index / Time")
# plt.ylabel("Value")
# plt.legend()
# plt.show()


##=====================.mat file
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ Load .mat file
# ==============================
file_path = "glass_train.mat"  # <-- replace with your .mat file path
mat_data = scipy.io.loadmat(file_path)

# ==============================
# 2️⃣ Inspect basic info
# ==============================
print("✅ Keys in .mat file:", mat_data.keys())

# Usually MATLAB adds '__header__', '__version__', '__globals__', so we skip them
data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
print("✅ Data keys:", data_keys)

# ==============================
# 3️⃣ Inspect each variable
# ==============================
for key in data_keys:
    var = mat_data[key]
    print(f"\n--- Variable: {key} ---")
    print("Type:", type(var))
    if isinstance(var, np.ndarray):
        print("Shape:", var.shape)
        print("First 5 entries:\n", var[:5])
        print("Min:", var.min())
        print("Max:", var.max())
        print("Mean:", var.mean())
        print("Std:", var.std())

# ==============================
# 4️⃣ Plot a variable (if numeric)
# ==============================
# Example: plot first variable
if data_keys:
    var = mat_data[data_keys[0]]
    if isinstance(var, np.ndarray):
        plt.figure(figsize=(10, 5))
        if var.ndim == 1:
            plt.plot(var)
            plt.title(f"{data_keys[0]} - 1D Data")
        elif var.ndim == 2:
            for i in range(min(var.shape[1], 5)):  # plot first 5 columns
                plt.plot(var[:, i], label=f"Col {i}")
            plt.title(f"{data_keys[0]} - 2D Data")
            plt.legend()
        elif var.ndim == 3:
            for i in range(var.shape[2]):
                plt.plot(var[0, :, i], label=f"Feature {i}")
            plt.title(f"{data_keys[0]} - 3D Data (first sample)")
            plt.legend()
        else:
            print("⚠️ Data has unexpected dimensions:", var.ndim)
        plt.xlabel("Index / Time")
        plt.ylabel("Value")
        plt.show()
