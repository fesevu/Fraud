# %%
# 2.1 PyTorch (CPU) – подмените индекс, если нужна CUDA
%pip install torch==2.6.0 torchvision torchaudio

# 2.2 PyG: с 2.3+ внешних библиотек почти нет, ставим одной строкой
%pip install torch_geometric 
%pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html

# 2.3 pyTigerGraph (понадобится torch заранее)
%pip install 'pyTigerGraph[gds]'

# 2.4 остальное
%pip install xgboost pandas matplotlib graphviz scikit-learn tqdm numpy
%pip install ipycytoscape

# 2.5 если drawSchema выдаёт ошибку “Graphviz executable not found”:
#   Ubuntu: sudo apt-get install graphviz
#   macOS:  brew install graphviz

# %% [markdown]
# # Detecting Fraudulent Accounts on the Ethereum Blockchain
# ### A Comprehensive Tutorial Using Graph Features and Machine Learning

# %% [markdown]
# ## Overview:
# In this tutorial, we delve into the exciting world of fraud detection on the Ethereum platform, focusing on transactions between accounts. Leveraging powerful tools like the `featurizer` in [pyTigerGraph](https://docs.tigergraph.com/pytigergraph/current/intro/), we engineer essential graph features from the transaction graph. To facilitate data retrieval, we employ the data loaders in pyTigerGraph, extracting valuable information from the TigerGraph database.
# 
# Our approach further incorporates [xgboost](https://xgboost.readthedocs.io/en/stable/), a gradient boost tree model, and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), enabling us to build powerful GNN (Graph Neural Network) models.
# 
# The dataset under study comprises transactions on the Ethereum platform, forming a transaction graph for Ether, the second-largest cryptocurrency. Wallets (i.e., accounts) on the platform serve as vertices in the graph, while edges represent transactions between these accounts. With 32,168 vertices and 84,088 edges, the dataset is derived from the publicly available Ethereum dataset from [XBlock](https://xblock.pro/ethereum#/dataset/13).
# 
# In this tutorial, we will explore various implementations of XGBoost and Graph Neural Networks (GNN) to achieve specific results. The covered topics include:
# 
# 1. XGBoost without Graph Features
# 2. XGBoost with Graph Features
# 3. XGBoost with Graph Features and FastRP embeddings
# 4. GNN with Graph Features and GraphSAGE
# 
# By following the step-by-step instructions provided for each method, we aim to obtain the desired outcomes:
# 
# | Model Type                                        | Accuracy        | Precision | Recall | Accuracy Improvement from Baseline |
# |---------------------------------------------------|-----------------|-----------|--------|------------------------------------|
# | XGBoost without Graph Features                    | 0.7737 (77.37%) | 0.1310    | 0.9865 | -                                  |
# | XGBoost with Graph Features                       | 0.9145 (91.45%) | 0.2846    | 0.9776 | + 14.08%                            |
# | XGBoost with Graph Features and FastRP embeddings | 0.9426 (94.26%) | 0.3737    | 0.9821 | + 2.81%                           |
# | GNN with Graph Features and GraphSAGE             | 0.9658 (96.58%) | 0.5035    | 0.6457 | + 2.32%                           |
# 
# Join us on this captivating journey as we explore the intricacies of detecting fraudulent accounts on the Ethereum blockchain using cutting-edge graph features and machine learning techniques.

# %% [markdown]
# ## Database Preparation
# ### Establishing a Connection to the TigerGraph Database
# To interact with the TigerGraph database, we will utilize the `TigerGraphConnection` class, which serves as the gateway for communication. This class encapsulates all the essential information required to establish and maintain a connection with the database. For more in-depth information on its functionalities, you can refer to the official [documentation](https://docs.tigergraph.com/pytigergraph/current/intro/).
# 
# To connect to your specific database, kindly make the necessary modifications in the `config.json` file that accompanies this notebook. In particular, ensure you set the value of `getToken` appropriately, depending on whether token authentication is enabled for your database. Note that token authentication is always enabled for tgcloud databases.
# 
# Let's proceed with the code implementation:

# %%
from pyTigerGraph import TigerGraphConnection
import json

# Read in DB configs
with open('../config.json', "r") as config_file:
    config = json.load(config_file)
    
conn = TigerGraphConnection(
    host=config["host"],
    username=config["username"],
    password=config["password"]
)

# %% [markdown]
# The provided Python code establishes a connection to the TigerGraph database. It reads the necessary configuration details from the `config.json` file and uses them to initialize the `TigerGraphConnection`. Once the connection is successfully established, we will be ready to perform various database tasks and interact with the data seamlessly.

# %% [markdown]
# ### Download Ethereum dataset and ingest it to the database.
# To enrich our analysis, we begin by downloading the Ethereum dataset, a crucial step in our project. Leveraging the `pyTigerGraph` library, we make use of the `Datasets` module to access the Ethereum dataset seamlessly. By instantiating the `Datasets` class with the parameter "Ethereum," we ensure that we obtain the appropriate dataset for our study. Subsequently, we proceed to ingest this dataset into the TigerGraph database through the `conn.ingestDataset()` function. To ensure a secure and authenticated ingestion process, we pass the required token using `getToken=config["getToken"]`. This essential operation forms the foundation for our subsequent analysis, enabling us to explore the Ethereum transaction graph and derive valuable insights for our fraud detection project.

# %%
from pyTigerGraph.datasets import Datasets
dataset = Datasets("Ethereum")
conn.ingestDataset(dataset, getToken=config["getToken"])

# %% [markdown]
# ### Visualize schema
# The next step in our project involves visualizing the schema of the TigerGraph database. To achieve this, we leverage the `pyTigerGraph` library's powerful visualization capabilities. By utilizing the `drawSchema` function from the `visualization` module, we obtain a clear and comprehensive representation of the database schema.

# %%
from pyTigerGraph.visualization import drawSchema
drawSchema(conn.getSchema(force=True))

# %% [markdown]
# ## Graph Feature Engineering
# 
# pyTigerGraph's featurizer empowers us with over 30 graph algorithms for essential feature calculations. Combining built-in **pagerank** and **FastRP** algorithms with **custom queries**, we derive valuable insights from the Ethereum transaction graph, laying the groundwork for our fraud detection analysis.

# %% [markdown]
# ### Initializing Featurizer Class
# The Python code `f = conn.gds.featurizer()` creates an instance of the `Featurizer` class from the `gds` module in the `pyTigerGraph` library. This line of code initializes a variable `f` that represents the featurizer object, allowing you to access and utilize the various graph algorithms and features provided by the `Featurizer` class.
# 
# With this instance, you gain the ability to apply a wide range of graph algorithms and feature calculations to the data within the TigerGraph database. These features can be used for further analysis, machine learning, or any other tasks that require insightful graph feature engineering.

# %%
f = conn.gds.featurizer()

# %% [markdown]
# ### PageRank
# The algorithm used in this code is [PageRank](https://docs.tigergraph.com/graph-ml/current/centrality-algorithms/pagerank), a classic graph algorithm that plays a crucial role in network analysis. In this particular implementation, `PageRank` is employed to rank and identify the most influential vertices (accounts) within the transaction graph (edges) of the Ethereum dataset. The algorithm takes parameters, such as the vertex type `("Account")`, edge type `("Transaction"`), and the desired result attribute `("pagerank")`, which indicates the specific attribute representing the `PageRank` score for each account. Additionally, the parameter `"top_k"` is set to 5, meaning the algorithm will return the top 5 accounts with the highest `PageRank` scores.
# 
# Upon executing the algorithm with these parameters, the results are stored in the variable `"results."` The code then uses the pandas library to normalize the results and create a tabular representation, allowing for easy analysis and interpretation of the top scores (highest PageRank) and their corresponding accounts. The output provides valuable insights into the most influential accounts in the Ethereum transaction graph, which can be instrumental in various analyses, including fraud detection and network analysis.

# %%
import pandas as pd

tg_pagerank_params = {
  "v_type": "Account",
  "e_type": "Transaction",
  "result_attribute": "pagerank",
  "top_k":5  
}
results = f.runAlgorithm("tg_pagerank", tg_pagerank_params)

pd.json_normalize(results[0]['@@top_scores_heap'])

# %% [markdown]
# ### Betweeness Centraility
# Next, lets utilize the algorithim [Betweenness Centrality](https://docs.tigergraph.com/graph-ml/current/centrality-algorithms/betweenness-centrality), a fundamental measure in graph theory that quantifies the importance of a vertex (account) within a network. Specifically, it calculates the number of shortest paths that pass through a given vertex, representing its influence in controlling the flow of information or transactions in the graph. 
# 
# In this implementation, `Betweenness Centrality` is applied to the Ethereum dataset's transaction graph to identify the most critical accounts that act as essential intermediaries in the flow of transactions. The algorithm's parameters are specified, including the vertex type set `("Account")`, the edge type set `("Transaction")`, and the reverse edge type `("reverse_Transaction")`, which allows for bi-directional traversal. The `"result_attribute"` is set to "betweenness," signifying that the algorithm will compute the betweenness centrality score for each account. Additionally, `"top_k"` is set to 5, indicating that the algorithm will return the top 5 accounts with the highest betweenness centrality scores. The results are stored in the variable "results" and are further presented in tabular format using pandas for easy analysis and interpretation, 

# %%
import pandas as pd

tg_pagerank_params = {
  "v_type_set": ["Account"],
  "e_type_set": ["Transaction"],
  "reverse_e_type": "reverse_Transaction",
  "result_attribute": "betweenness",
  "top_k":5  
}
results = f.runAlgorithm("tg_betweenness_cent", tg_pagerank_params)

pd.json_normalize(results[0]['top_scores'])

# %% [markdown]
# ### Weakly Connected Components
# Next, lets utilize the algorithim [Weakly Connected Components](https://docs.tigergraph.com/graph-ml/current/community-algorithms/connected-components), a fundamental algorithm in graph theory that determins communities of vertices (accounts) within a network. 
# 
# A connected component is the maximal set of connected vertices inside a graph, plus their connecting edges. Inside a connected component, you can reach each vertex from each other vertex.
# 
# Graph theory distinguishes between strong and weak connections:
# 
# * A subgraph is strongly connected if it is directed and if every vertex is reachable from every other following the edge directions.
# 
# * A subgraph is weakly connected if it is undirected or if the only connections that exist for some vertex pairs go against the edge directions.
# 
# 
# 
# In this implementation, `Weakly Connected Components` is applied to the Ethereum dataset's transaction graph to identify the most critical accounts that act as essential intermediaries in the flow of transactions. The algorithm's parameters are specified, including the vertex type set `("Account")`, the edge type set `("Transaction")`, and the reverse edge type `("reverse_Transaction")`, which allows for bi-directional traversal. The `"result_attribute"` is set to "wcc_id", which will store the results as an attribute on each Account vertex.
# 
# We will then use the `component_size` query to compute the feature we will use - the size of the connected component that each Account belongs to.

# %%
tg_wcc_params = {
    "v_type_set": ["Account"],
    "e_type_set": ["Transaction", "reverse_Transaction"],
    "result_attribute": "wcc_id"
}

results = f.runAlgorithm("tg_wcc", params=tg_wcc_params)

# %%
f.installAlgorithm("component_size", query_path="./gsql/component_size.gsql")
f.runAlgorithm("component_size", params={"result_attr": "wcc_size"}, custom_query=True, schema_name=["Account"], feat_type="INT", feat_name="wcc_size")

# %% [markdown]
# ### Degree Features
# 
# The following code demonstrates the computation of Degree Features within the graph structure. Firstly, it installs the "degrees" algorithm using `f.installAlgorithm("degrees", query_path="./gsql/degrees.gsql")`, which prepares the algorithm for execution. Next, it runs the "degrees" algorithm through `f.runAlgorithm("degrees", custom_query=True)`, triggering the computation process.

# %%
f.installAlgorithm("degrees", query_path="./gsql/degrees.gsql")
f.runAlgorithm("degrees", custom_query=True)

# %% [markdown]
# #### Additional information about degrees.gsql
# The algorithm's core logic is described within the custom GSQL query titled "degrees." This query calculates both the in-degree and out-degree of each vertex (account) in the graph. The algorithm utilizes the `SumAccum` function to accumulate the in-degree and out-degree values for each vertex. The query traverses the graph, identifying all the incoming and outgoing edges for each vertex, and increments the corresponding degree counters accordingly.
# 
# Upon calculating the degree values, the algorithm sets the attributes "in_degree" and "out_degree" for each vertex, capturing the calculated values. Finally, the algorithm prints a success message, indicating the successful computation of the Degree Features.
# 
# In summary, the code `degrees.gsql` prepares, executes, and computes Degree Features within the graph, effectively capturing the in-degree and out-degree values for each vertex, providing valuable insights into the connections and influence of each account in the Ethereum transaction graph.

# %% [markdown]
# ### Amount Features
# 
# The provided code pertains to the computation of Amounts Features within the graph. Firstly, it installs the "amounts" algorithm using `f.installAlgorithm("amounts", query_path="./gsql/amounts.gsql")`, ensuring the algorithm is prepared for execution. Subsequently, the "amounts" algorithm is run through `f.runAlgorithm("amounts", custom_query=True)`, initiating the process of calculating Amounts Features.

# %%
f.installAlgorithm("amounts", query_path="./gsql/amounts.gsql")
f.runAlgorithm("amounts", custom_query=True)

# %% [markdown]
# #### Additional Information about amount.gsql
# The algorithm's core logic is defined within the custom GSQL query titled "amounts.gsql." This query performs calculations related to sending and receiving amounts for each vertex (account) in the graph. Four accumulators are employed, namely `@send_min`, `@send_amount`, `@recv_min`, and `@recv_amount`, to store the minimum and sum of amounts sent and received.
# 
# The query traverses the graph and identifies all transactions between accounts, considering both outgoing and incoming edges. It accumulates the amounts of transactions, updating the corresponding accumulators for each vertex.
# 
# Upon computing the amounts, the algorithm sets the attributes "recv_min," "recv_amount," "send_min," and "send_amount" for each vertex, capturing the calculated values. Additionally, it handles the special case where a vertex has no incoming or outgoing edges (i.e., in_degree or out_degree equals zero), setting the corresponding "recv_min" or "send_min" to zero in such cases.
# 
# Finally, the algorithm prints a success message, confirming the successful computation of the Amounts Features.
# 
# In summary, the `amounts.gsql` code prepares, executes, and computes Amounts Features within the graph, calculating the minimum and sum of amounts sent and received for each account in the Ethereum transaction graph. This process allows for a detailed understanding of the transaction patterns and amounts associated with each vertex, contributing to further analysis and insights in various applications, including fraud detection and network analysis.

# %% [markdown]
# ### FastRP Embeddings

# %%
params={"v_type_set": ["Account"],
        "e_type_set": ["Transaction", "reverse_Transaction"],
        "output_v_type_set": ["Account"],
        "iteration_weights": "1,2,4",
        "beta": -0.1,
        "embedding_dimension": 128,
        "embedding_dim_map": [],
        "default_length": 128,
        "sampling_constant": 3,
        "random_seed": 42,
        "component_attribute": "",
        "result_attribute": "embedding",
        "choose_k": 0}

f.runAlgorithm("tg_fastRP", params=params)

# %% [markdown]
# ### Check Labels
# Next, lets check and analyze the labels of accounts in a database. Firstwe will use the connection object named "conn" to interact with the database. The code first queries the database to count the number of accounts labeled as frauds (with "is_fraud = 1") and the number of accounts labeled as non-frauds (with "is_fraud = 0"). It then calculates the percentage of fraud and non-fraud accounts out of the total number of accounts and prints the results. The output displays the count and percentage of both fraud accounts and normal accounts in the database, providing valuable insights into the distribution of labeled accounts.

# %%
frauds = conn.getVertexCount("Account", "is_fraud = 1") 
nonfrauds = conn.getVertexCount("Account", "is_fraud = 0") 
print("Fraud accounts: {} ({:.2f}%%)".format(frauds, frauds/(frauds+nonfrauds)*100))
print("Normal accounts: {} ({:.2f}%%)".format(nonfrauds, nonfrauds/(frauds+nonfrauds)*100))

# %% [markdown]
# ## Graph Neural Network
# 
# Moving on let's define the hyperparameters for a Graph Neural Network (GNN) model and its corresponding training environment. The hyperparameters include a batch size of 1024, the number of neighbors to consider (num_neighbors) during message passing, the number of hops (num_hops) for message propagation in the graph, a hidden dimension of 256 for the GNN layers, two GNN layers (num_layers), and a dropout rate of 0.25 for regularization. These hyperparameters play a crucial role in shaping the GNN's architecture and controlling its training process. Properly tuning these hyperparameters can significantly impact the model's performance on graph-related tasks, such as node classification or link prediction.

# %%
hp = {
    "batch_size": 1024, 
    "num_neighbors": 200, 
    "num_hops": 2, 
    "hidden_dim": 256, 
    "num_layers": 2, 
    "dropout": 0.25
}

# %% [markdown]
# ### Create Neighbor Loaders
# 
# Next, let's create the Neighbor Loaders used for training and validation in the context of a Graph Neural Network (GNN) model. The neighbor loaders are an essential component in graph-based machine learning tasks as they handle the efficient retrieval of neighboring node information for each target node during training and validation. The `conn.gds.neighborLoader` function is used to create these loaders, specifying various parameters. The "v_in_feats" parameter lists the input features of the target nodes, such as in-degree, out-degree, send_amount, send_min, recv_amount, recv_min, pagerank, and betweenness. The "v_out_labels" parameter designates the output labels for the target nodes, specifically "is_fraud" in this case. The "v_extra_feats" parameter includes additional features related to the target nodes, namely "is_training" and "is_validation" labels. The "output_format" is set to "PyG" to align the data format with the popular PyTorch Geometric (PyG) library. Other parameters, such as batch size, number of neighbors, number of hops, filtering, shuffling, and timeout, are specified to control the data loading process. By creating these Neighbor Loaders, the GNN model can efficiently access and utilize neighboring node information during its training and validation phases, significantly enhancing its performance in graph-related tasks.

# %%
with open('../config.json', "r") as config_file:
    config = json.load(config_file)
    
conn = TigerGraphConnection(
    host=config["host"],
    username=config["username"],
    password=config["password"],
    graphname="Ethereum"
)

train_loader, valid_loader = conn.gds.neighborLoader(
    v_in_feats=["in_degree","out_degree","send_amount","send_min","recv_amount","recv_min","pagerank","betweenness"],
    v_out_labels=["is_fraud"],
    v_extra_feats=["is_training", "is_validation"],
    output_format="PyG",
    batch_size=hp["batch_size"],
    num_neighbors=hp["num_neighbors"],
    num_hops=hp["num_hops"],
    filter_by = ["is_training", "is_validation"],
    shuffle=True,
    timeout=600000
)

# %% [markdown]
# ### Create Graph Attention Network
# 
# Next, a Graph Attention Network (GAN) is created for vertex classification using the GraphSAGE model. The GAN is designed to perform vertex classification, where each vertex in the graph is assigned to one of two classes. The code imports the required GraphSAGE module from the pyTigerGraph library and initializes the GAN with specific hyperparameters. The "num_layers" parameter controls the number of layers in the GAN, while the "out_dim" parameter sets the output dimension of the GAN's classification layer, which is two in this case, representing the two classes. The "hidden_dim" parameter determines the dimension of the hidden layers in the model. The "dropout" parameter introduces dropout regularization to prevent overfitting during training. Additionally, the "class_weights" parameter is used to address class imbalance, setting a higher weight for the minority class (class 1) with a weight ratio of 1:15. By creating this Graph Attention Network, the model can effectively learn to classify vertices in the graph, particularly in tasks with imbalanced class distributions.

# %%
from pyTigerGraph.gds.models.GraphSAGE import GraphSAGEForVertexClassification
import torch

gs = GraphSAGEForVertexClassification(num_layers = hp["num_layers"],
                                      out_dim = 2,
                                      hidden_dim = hp["hidden_dim"],
                                      dropout = hp["dropout"],
                                      class_weights = torch.FloatTensor([1, 15]))

# %% [markdown]
# ### Train Model
# Now we will train the previously defined Graph Attention Network (GAN) using the "gs.fit" method. The GAN is being trained on the provided "train_loader" and validated using the "valid_loader" for a total of 10 epochs. This process involves optimizing the model's parameters and learning the best representation to classify vertices in the graph accurately.

# %%
gs.fit(train_loader, valid_loader, number_epochs=10)

# %%
final_metrics = gs.trainer.get_eval_metrics()

# %% [markdown]
# ### Model Evaluation
# The final evaluation metrics of the trained Graph Attention Network (GAN) are obtained and presented as a dictionary. The evaluation results include an **accuracy of 96.62%**, precision of 54.68%, and recall of 63.07%. Additionally, the confusion matrix showcases the number of true positive (152), false positive (126), true negative (5996), and false negative (89) instances. The calculated loss for the GAN is 26.59. These metrics provide insights into the GAN's performance in accurately classifying vertices in the graph and highlight the effectiveness of the model in handling imbalanced class distributions.

# %%
final_metrics

# %% [markdown]
# Let's print the final evaluation metrics (accuracy, precision, and recall) for the Graph Attention Network (GNN) model, providing a clear and concise summary of its performance in vertex classification.

# %%
# Print the final performance metric for the GNN model
print("Accuracy {:.4f}, Precision {:.4f}, Recall {:.4f}".format(
    final_metrics["accuracy"], final_metrics["precision"], final_metrics["recall"]))

# %% [markdown]
# ### Visualize Results with Confusion Matrix
# 
# We will create visual representation of the confusion matrix based on the evaluation metrics. The confusion matrix is displayed as a heatmap using the "matshow" function from Matplotlib, where different colors represent the number of true positive, false positive, true negative, and false negative instances. The values within the heatmap are annotated using "text" to show the exact counts of each class prediction. The plot is accompanied by labeled axes, and the title "Confusion Matrix" is added to provide a clear understanding of the model's performance in distinguishing between actual and predicted classes. The visualization of the confusion matrix aids in assessing the model's classification accuracy and identifying any potential misclassifications.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(final_metrics["confusion_matrix"], cmap=plt.cm.Blues, alpha=0.8)

for i in range(final_metrics["confusion_matrix"].shape[0]):
    for j in range(final_metrics["confusion_matrix"].shape[1]):
        ax.text(x=j, y=i,s=final_metrics["confusion_matrix"].values[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# %% [markdown]
# ### Explain Model

# %% [markdown]
# Let's sample a random vertex that the model predicts to be fraudulent and retrieve the data representing the subgraph surrounding that vertex. The data is fetched from the "train_loader" using the "fetch" method with specific attributes, such as the vertex type "Account" and the primary ID "0x5093c4029acab3aff80140023099f5ec6ca7d52f." This process allows for a closer examination of the subgraph and gaining insights into the model's predictions and the underlying patterns associated with fraudulent vertices.

# %%
from torch_geometric.explain import Explainer, GNNExplainer

data = train_loader.fetch([{"type": "Account", "primary_id": "0x5093c4029acab3aff80140023099f5ec6ca7d52f"}])

explainer = Explainer(
    model=gs.model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='raw',  # Model returns log probabilities.
    ),
)

# %%
data

# %% [markdown]
# Next let's us an "explainer" object to generate explanations for a Graph Attention Network (GAN) model's predictions in vertex classification. It visualizes the feature importance scores of the input features and creates a graphical representation of the subgraph around a sampled vertex to gain insights into the model's decision-making process. The explanations help in understanding critical features and validating the model's performance.

# %%
explanation = explainer(data.x.float(), data.edge_index)

explanation.visualize_feature_importance(top_k=len(train_loader.v_in_feats), feat_labels=train_loader.v_in_feats)

explanation.visualize_graph()

# %% [markdown]
# ## Conclusion

# %% [markdown]
# * Graph features significantly enhance the performance of traditional ML models, as observed in the XGBoost example, where PageRank and the size of the connected components emerge as important features for classification.
# * The Graph Attention Network (GAT) outperforms XGBoost in terms of accuracy and precision, demonstrating its superiority in vertex classification tasks.
# * Although GAT achieves slightly lower recall compared to XGBoost, it still maintains a good level of performance, making it a promising choice for various graph-related applications, particularly in scenarios with imbalanced class distributions.
# 
# ### Accuracy Improvement
# The machine learning model performance shows significant improvements when incorporating graph features and advanced techniques. Comparing the different models, the XGBoost model with graph features demonstrates a notable accuracy increase from 0.7737 **(77.37% accuracy)** to 0.9145 **(91.45% accuracy)**, while the addition of FastRP embeddings further boosts the accuracy to 0.9426 **(94.26% accuracy)**. However, the Graph Neural Network (GNN) model with GraphSAGE surpasses all XGBoost variations, achieving an impressive accuracy of 0.9658 **(96.58% accuracy)**. Although the GNN model shows slightly lower recall compared to some XGBoost variants, its overall performance, especially in accuracy and precision, makes it a superior choice for vertex classification tasks involving complex graph data. The results underscore the benefits of leveraging graph features and GNNs to achieve more accurate and sophisticated predictions in graph-based machine learning tasks.

# %%
tick_labels = ["Accuracy", "Precision", "Recall"]
x_tree_base = [metrics['acc_tree_base'][-1], metrics['prec_tree_base'][-1], metrics['rec_tree_base'][-1]]
x_tree_graph = [metrics['acc_tree_graph'][-1], metrics['prec_tree_graph'][-1], metrics['rec_tree_graph'][-1]]
x_gat = [final_metrics['accuracy'], final_metrics['precision'], final_metrics['recall']]
x_fastrp_tree = [metrics['acc_fastrp_tree'][-1], metrics['prec_fastrp_tree'][-1], metrics['rec_fastrp_tree'][-1]]
y = np.arange(len(tick_labels))
bar_width = 0.2
plt.barh(y-bar_width*1.5, x_tree_base, bar_width, label="Baseline")
plt.barh(y-bar_width/2, x_tree_graph, bar_width, label="+Graph features")
plt.barh(y+bar_width/2, x_fastrp_tree, bar_width, label="+FastRP")
plt.barh(y+bar_width*1.5, x_gat, bar_width, label="GAT")
plt.yticks(y, tick_labels)
plt.legend()
plt.gca().invert_yaxis()
plt.xlabel("Metric value")

# %%



