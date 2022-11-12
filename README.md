[![image.png](https://i.postimg.cc/cHQHGRYD/image.png)](https://postimg.cc/N9jY1rdT)
📕 Predict a publication trend of AI journals / conferences using GNNs  
Baseline paper: <a href="https://ieeexplore.ieee.org/document/9054769">Structured Citation Trend Prediction Using Graph Neural Network</a>
</br>

## Members
 👑차지수<br>
 윤수진<br>
 조현우<br>
 진현빈<br>
 박수빈<br>
 김산<br>
 김민서<br>
</br>

## Requirements

### Verisions *(Recommended)*
* Python 3.7.x  
* Pytorch 1.12.1+cu113  
* Torch_geometric 2.1.0  

### Docker
We recommend using our [Dockerfile](https://github.com/hwCh00/graph-dive/blob/main/Dockerfile) to get started easily
```
## build docker image
$ docker build -t graph-dive:latest . 

## execute docker container
$ docker run --name graph-dive --ipc=host -it -v <working_dir>:/workspace -w /workspace graph-dive:latest /bin/bash
```
</br>

## Model
We follow the architecture of baseline paper which is based on GATs and GCNs.  
**[Training stage]**
![train_figure](https://user-images.githubusercontent.com/96547408/201436402-ab75bdf4-a249-4c15-82db-d6496a480fde.jpg)


**[Prediction stage]**
![test_figure](https://user-images.githubusercontent.com/96547408/201436425-743df10e-c2a7-4424-9141-f39611579f07.jpg)
</br>

## Dataset
### MAG(Microsoft Academic Graph)
We use author, affiliation, the number of citation, title and abstract of paper, year as raw inputs.
![MAG schema](https://user-images.githubusercontent.com/96547408/201435997-98326513-dfcb-4d05-bec1-90a30177e152.png)
Please check <a href="https://learn.microsoft.com/en-us/academic-services/graph/reference-data-schema">this webpage</a> for more information. 

### Data directory tree
Directory tree including data should be as follows:  
``` 
├─graph-dive/
└─data/
	├─ affiliationembedding.csv
	├─ edge_data/
	│   ├─ 1158167855_refs.csv #{CVPR_conference_id}_refs.csv
	│   ├─ 1184914352_refs.csv #{AAAI_conference_id}_refs.csv
	│   └─ ...
	├─ year_data/
	│   ├─ 1158167855.csv #{CVPR_conference_id}.csv
	│   ├─ 1184914352.csv #{AAAI_conference_id}.csv 
	│   └─ ...
	├─ json_1158167855/ # CVPR
	│   ├─ {paper_id1}.json
	│   ├─ {paper_id2}.json
	│   └─ ...
	├─ json_1184914352/ # AAAI
	│   └─ ...
	...
```

For each journal/conference, conference IDs are look like:  
|Conference|Conference ID|# of nodes|
|------|-------|------|
|ICML|1180662882|8653|
|ICASSP|1121227772|16997|
|NeurIPS|1127325140|8113|
|AAAI|1184914352|13766|
|EMNLP|1192655580|5667|
|CVPR|1158167855|13058|
|ICDM|1183478919|4169|
|CIKM|1194094125|4201|
</br>

## Run
Command examples
```
# CVPR
$ bash scripts/run_CVPR.sh

# ICASSP
$ bash scripts/run_ICASSP.sh
```
Note that the number of valid data are smaller than the values stated above due to insufficient sources(OpenAlex API, MAG dataset, etc..)

## 📝 SKILLS
Frameworks:  
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/scikit-learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/> <img src="https://img.shields.io/badge/pyg-3C2179?style=flat-square&logo=pyg&logoColor=white"/>
