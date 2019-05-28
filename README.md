# GAPNet:Graph Attention based Point Neural Network for Exploiting Local Feature of Point Cloud
created by Can Chen, Luca Zanotti Fragonara, Antonios Tsourdos from Cranfield University

[[Paper]](https://arxiv.org/abs/1905.08705)

# Overview
We propose a graph attention based point neural network, named GAPNet, to learn shape representations for point cloud. Experiments show state-of-the-art performance in shape classification and semantic part segmentation tasks.

In this repository, we release code for training a GAPNet classification network on ModelNet40 dataset and a part segmentation network on ShapeNet part dataset.

# Requirement
* [TensorFlow](https://www.tensorflow.org/)

# Point Cloud Classification
* Run the training script:
``` bash
python train.py
```
* Run the evaluation script after training finished:
``` bash
python evaluate.py --model=network --model_path=log/epoch_185_model.ckpt
```

# Point Cloud Part Segmentation
* Run the training script:
``` bash
python train_multi_gpu.py
```
* Run the evaluation script after training finished:
``` bash
python test.py --model_path train_results/trained_models/epoch_130.ckpt
```

# Citation
Please cite this paper if you want to use it in your work.

``` bash
@article{chen2019gapnet,
  title={GAPNet: Graph Attention based Point Neural Network for Exploiting Local Feature of Point Cloud},
  author={Chen, Can and Fragonara, Luca Zanotti and Tsourdos, Antonios},
  journal={arXiv preprint arXiv:1905.08705},
  year={2019}
}
```

# License
MIT License

