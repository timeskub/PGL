## Improved Sort Pooling Graph Neural Networks for Link Prediction

This experiment is based on stanford OGB benchmark. The description of 《Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification》 is [avaiable here](https://arxiv.org/pdf/2009.03509.pdf). The steps are:

### Note!
We propose **UniMP_large**, where we extend our base model's width by increasing ```head_num```, and make it deeper by incorporating [APPNP](https://www.in.tum.de/daml/ppnp/) . Moreover, we firstly propose a new **Attention based APPNP** to further improve our model's performance.


### Install environment:
``` 
!pip install torch==1.7.0+cu101 torchvision>=0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install torch-geometric==1.6.1
!pip install ogb==1.2.4
    
```
### DRNL+MDGCNN:
   
  ```Jupyter file - main.ipynb``` to get the baseline of DRNL+MDGCNN
### The **detailed hyperparameter** is:
```  
args = {
        'runs':5,
        'dataset':'ogbl-collab', 'use_valedges_as_input':True, 'num_hops': 1, 'use_feature':True, 'use_edge_weight': False, 
        'train_percent':10, 'val_percent':100,'test_percent':100, 'node_label':'drnl', 'ratio_per_hop': 1, 'max_nodes_per_hop':None,
        'batch_size':32, 'num_workers':16,c'hidden_channels':32, 'model':'DGCNN', 'num_layers': 3,
        'pretrained_node_embedding':False,'train_node_embedding':False,'max_z':1000, 'lr': 0.0001, 'modified_sortpooling': True,
        'epochs':50, 'sortpool_k':0.6, 'dynamic_train':False, 'dynamic_val':False, 'dynamic_test':False,
        }
```  


### DRNL+SAGE:
   
  ```Jupyter file - main.ipynb``` to get the baseline of DRNL+MDGCNN
### The **detailed hyperparameter** is:
```  
args = {
        'runs':5,
        'dataset':'ogbl-collab', 'use_valedges_as_input':True, 'num_hops': 1, 'use_feature':True, 'use_edge_weight': False, 
        'train_percent':10, 'val_percent':100,'test_percent':100, 'node_label':'drnl', 'ratio_per_hop': 1, 'max_nodes_per_hop':None,
        'batch_size':32, 'num_workers':16,c'hidden_channels':32, 'model':'SAGE', 'num_layers': 3,
        'pretrained_node_embedding':False,'train_node_embedding':False,'max_z':1000, 'lr': 0.0001, 'modified_sortpooling': False,
        'epochs':15, 'sortpool_k':0.6, 'dynamic_train':False, 'dynamic_val':False, 'dynamic_test':False,
        }
```  



### Reference performance for OGB:

| Model              |Test Accuracy    | Parameters    | Hardware |
| ------------------ |--------------   | -------------- |----------|
| DRNL+MDGCNN     | 0.7225  ± 0.0015  | 105,282  | Tesla P100-PCIE-16GB |
| DRNL+SAGE        | 0.7311  ± 0.0021 | 47,521 | Tesla P100-PCIE-16GB |
   
