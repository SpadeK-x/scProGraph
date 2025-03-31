# scProGraph  

### **Brief Introduction**  

scProGraph is an innovative prototype-guided graph neural network framework designed for cell type annotation and functional gene subgraph discovery in single-cell RNA sequencing (scRNA-seq) data. By jointly modeling cell-cell similarity graphs and gene-gene interaction networks, this method employs a bagging strategy to mitigate noise interference while integrating Graph Transformer architectures and prototype learning mechanisms to simultaneously optimize classification performance and biological interpretability. Experimental results demonstrate that scProGraph achieves state-of-the-art performance across multiple disease datasets (including leukemia, breast cancer, and colorectal cancer), with an average accuracy exceeding 90% (peaking at 99.75%). Notably, the extracted gene subgraphs exhibit significant coverage of known protein-protein interaction networks (e.g., 26.92% for macrophage-related subgraphs), validating their biological relevance. Beyond serving as a high-precision annotation tool, scProGraph provides novel insights for biomarker discovery and gene regulatory mechanism exploration.

---

##  **Installation**  
### Requirements 
```
- python                    3.9.0             
- torch-geometric           2.4.0
```
you can install requirements by using:
```
pip install -r requirements.txt 
```

##  **Usage**  

Firstly, you need to convert rawdata into a format that the model can input:
* expression matrix: rows denote genes, and cols denote cells.
* cell type data: cell type of each cell.

you can do this by running: 
```
python data_precrocess.py
```

then train the model by runing:
```
python -m models.train_gnns --clst * --sep * --file * --id1 * --id2 *
```

Notes:

* \* is suitable parameter
* clst and sep are coefficient for clst loss and sep loss
* file are formated as {species}\_{tissue}{num}\_{data / celltype}.csv
* id1 and id2 are the train and test num

train the supervised learning model by runing:
```
python -m models.group --file * --id1 * --id2 *
```

predict with the test dataset by running:
```
python -m models.predict --file * --id1 * --id2 *
```

##  **Project Structure** 
```
scProGraph/  
├── datasets/
│   └── SingleCellDataset/  
│       ├── train/    
│       └── test/  
├── models/
│   ├── GCN    
│   └── ...  
├── Configure.py                    # Configuration files
├──  ...
└── README.md                       # This file  
```

---

## Results


| Model          |  Accuracy   | Precision |   Recall    | F1-score  |
|:---------------|:-----------:|:---------:|:-----------:|:---------:|
| CHETAH         |   70.67%    |   54.83   |    42.93    |   45.82   |
| treeArches     |   71.68%    |   49.95   |    45.75    |   44.58   |
| scANVI         |   89.35%    |   89.05   |  **88.91**  |   88.19   |
| sciBet         |   86.71%    |   86.09   |    85.45    |   84.53   |
| scLearn        |   72.43%    |   58.62   |    52.29    |   53.06   |
| scPoli         |   52.57%    |   50.70   |    51.80    |   49.14   |
| scPred         |   85.24%    |   67.13   |    63.83    |   64.44   |
| scType         |   61.68%    |   45.09   |    57.40    |   47.56   |
| **scProGraph** | **90.47%**  | **89.51** |    88.76    | **88.33** |


## Explainability
scProGraph is capable of effectively capturing latent relationships among genes through the construction of subgraphs. In the projection stage, the extracted graph structure is mapped as gene-gene subgraphs. By analyzing the structural information of these subgraphs, one can elucidate the interaction relationships among genes in different cell types and their roles in various biological processes.


Figure below presents the generation results of selected gene subgraphs when applying this method to the E-MTAB-8107(2) dataset. In these subgraphs, each node represents a gene, and the connecting edges indicate interactions or correlations between genes. Each column contains two subgraphs, corresponding to the same type of subgraph structure.


We employed the Protein-Protein Interaction network data collected by Michelle M. Li to systematically compare the PPI networks of Macrophage, Fibroblast, and Monocyte cells in dataset E-MTAB-8107(2). 
![file:///src/fig1.png](https://github.com/SpadeK-x/scProGraph/blob/main/src/fig1.png))
![file:///src/fig2.png](https://github.com/SpadeK-x/scProGraph/blob/main/src/fig2.png))
