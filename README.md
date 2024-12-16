# Proof-of-Concept

GAT-Enhanced DGNN for Financial Fraud Detection
Description
This project implements a Graph Attention Network (GAT) enhanced Dynamic Graph Neural Network (DGNN) for detecting financial fraud. The system models relationships between transaction nodes (accounts) over time, allowing it to capture dynamic and evolving fraud patterns. By focusing on graph structures, it detects sophisticated fraud schemes like money laundering, phishing, and pyramid schemes.

Data
The dataset consists of financial transaction records with the following key features:

Transaction ID: Unique identifier for each transaction.
Source Account: The account that initiates the transaction.
Destination Account: The recipient of the transaction.
Transaction Amount: The value of the transaction.
Timestamp: The date and time of the transaction.
Fraud Label: Indicates whether the transaction is fraudulent (1) or legitimate (0).
Data preprocessing involves cleaning missing values, constructing graph-based relationships between accounts, and scaling transaction features.

Code Structure
The project is organized as follows:

 
├── data_preprocessing.py     # Cleans and prepares the transaction data
├── graph_construction.py     # Builds graph structures (nodes = accounts, edges = transactions)
├── gat_dgnn_model.py         # Defines the GAT-enhanced DGNN model architecture
├── train.py                  # Handles model training
├── evaluate.py               # Evaluates model performance (AUC, precision, recall)
├── visualization.py          # Plots confusion matrix, PCA, AUC-ROC, and other visualizations
├── utils.py                  # Utility functions (logging, metrics, etc.)
└── GAT-enhanced-DGNN.ipynb   # Jupyter Notebook containing the complete process
Technologies Used
Programming Language: Python
Machine Learning Libraries: PyTorch, NumPy, Scikit-learn
Graph Processing: NetworkX, PyTorch Geometric (PyG)
Visualization: Matplotlib, Seaborn
This project applies advanced graph neural network (GNN) techniques for fraud detection, with a special focus on attention mechanisms to prioritize the most relevant transaction nodes and edges.
