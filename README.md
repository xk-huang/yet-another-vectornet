# Reimplement VectorNet :car:

Paper: [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259)

Still under construction:

- [x] finish the feature preprocessor
- [x] finish the hierarchical GNN
- [x] overfit the tiny sample dataset
- [x] batchify the data and compute subgraph in parallel
- [X] evaluate results on DE / ADE metrics
- [x] refine the feature preprocessor (how to encode the features)
- [x] Check the correctness of hierarchical-GNN's implementation
- [x] run on the whole dataset (running)
- [x] add multi-GPU training (currently too slow, 2h an epoch)
- [ ] add uni-test for each modules
- [ ] More advanced trajectory predictor, generate diverse trajectories (MultiPath, or variational RNNs; current using MLP)
- [ ] add node feature completing module


Inplement a Vectornet: hierarchical GNN encoder (no feature completing) + MLP predictor, without node feature completing.

~~The performance on test is 3.255 on  minADE (K=1) v.s that in paper of 1.81.~~ (bug found in `GraphDataset`: the former implementation contained *self-loops connection* in graph data, which was wrong; and the preprocessed `dataset.pt` was also wrong; now the model is still trainning...)

After I fix the bug about self-loops in `Graph.Data`, I re-train the network with the same setting but only to find the performance on the validation set remains the same for about 2.6 of ADE, which was so disappointing. Notice that I only use the context (social + lanes) with about 5-10 meters around each agent (not enough machine for me), so I tried to change the context radius to 100 meters in `config.py` file (in the paper it's 200 * 200 if my memory serves me right). Unfortunately, the machines in the lab are not accessible to me right now, so I couldn't train the network with these new settings. :cry:

branch `master` is sync with branch `large-scale`; branch `overfit-small` is archived.


---

## Table of Contents

- [Environment](#Environment)
- [Usage](#Usage)
- [Results on val and test](#Results-on-val-and-test)
- [Result and visualization for overfitting tiny dataset](#Result-and-visualization-for-overfitting-tiny-dataset)

---

## Environment

Multi-GPU training on Windows Serer 2016; CUDA version 10.1; 2 Titan Xp GPUs.

Install the packages mentioned in requirements.txt
```
pip install -r requirements.txt
```

> torch==1.4.0, 
argoverse-api, 
numpy==1.18.1, 
pandas==1.0.0, 
matplotlib==3.1.1, 
torch-geometric==1.5.0

## Usage

For pre-processed data, pre-trained model, and results `*.h5` file: [Google Drive](https://drive.google.com/drive/folders/1XJ2Oz4Qc2UstnfRw3DNvQThuEVvM6tUL?usp=sharing)

(Remember to run `find . -name "*.DS_Store" -type f -delete` if you're using MacOS)

0) Install [Argoverse-api](https://github.com/argoai/argoverse-api/tree/master/argoverse). Download `HD-maps` in argoverse-api as instructed.

1) download [the prepared dataset objects on Google Drive](https://drive.google.com/drive/folders/1XJ2Oz4Qc2UstnfRw3DNvQThuEVvM6tUL?usp=sharing) directly and unzip it in path `.`, and skip step 3.

    or prepared the dataset (batchify ...) from raw *.csv. 
       
    put all data (folders named `train/val/test` or a single folder `sample`) in `data` folder.
    
    An example folder structure:
    ```
    data - train - *.csv
         \        \ ...
          \
           \- val - *.csv
            \       \ ...
             \
              \- test - *.csv
                       \ ...
    ```
2) Modify the config file `utils/config.py`. Use the proper env paths and arguments.

3) Feature preprocessing, save intermediate data input features (compute_feature_module.py)
    ```
    $ python compute_feature_module.py
    ```
    Use (200, 200) size for a single sequence as the paper told.

4) Train the model (`train.py`; overfit a tiny dataset by setting `small_dataset = True`, and use `GraphDataset` in `dataset.py` to batchify the data)
    ```
    $ python train.py
    ```

---

## Results on val and test

Some predicting results were uploaded to the Argoverse contest, check the board via the [url](https://evalai.cloudcv.org/web/challenges/challenge-page/454/leaderboard/)

Submission ID of the repo: @xkhuang

### Result on val


| model params                                                 | minADE (K=1) | minFDE (K=1) |
| ------------------------------------------------------------ | ------------ | ------------ |
| results in paper | 1.66  | 3.67  |
| epoch_24.valminade_2.637.200624.xkhuang.pth                  | 2.637        |              |

### Result on test

| model params                                                 | minADE (K=1) | minFDE (K=1) |
| ------------------------------------------------------------ | ------------ | ------------ |
| results in paper | 1.81  | 4.01  |
| epoch_24.valminade_2.637.200624.xkhuang.pth                  | 3.255298     | 6.992046     |


---

## Result and visualization for overfitting tiny dataset

Sample results are shown below:
* red lines are agent input and ground truth output
* blue points are predicted feature tarjectory
* light blue lanes are other moving objects
* grey lines are lanes

### Using nearby context (about 5M around):
| | |
|:-------------------------:|:-------------------------:|
| ![](images/1.png) | ![](images/2.png) |
| ![](images/3.png) | ![](images/4.png) |

### Using 200 * 200 context (about 100M around):
with lanes:
| | |
|:-------------------------:|:-------------------------:|
| ![](images/200*200-1-1.png) | ![](images/200*200-2-1.png) |
| ![](images/200*200-3-1.png) | ![](images/200*200-4-1.png) |
| ![](images/200*200-5-1.png) |  |

without lanes:
| | |
|:-------------------------:|:-------------------------:|
| ![](images/200*200-1-2.png) | ![](images/200*200-2-2.png) |
| ![](images/200*200-3-2.png) | ![](images/200*200-4-2.png) |
| ![](images/200*200-5-2.png) |  |