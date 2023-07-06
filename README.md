# DPCL 
Domain Generalization Semantic Segmentation

<img src="./dpcl_framework.png"></img>

## DPCL - Pytorch

Implementation of <a href="https://doi.org/10.1609/aaai.v37i9.26280">DPCL</a>, domain generalized network for segmentation, in Pytorch. The full paper can be found in <a href="https://arxiv.org/abs/2303.01906">

### How to Run RobustNet
We evaludated DPCL on [Cityscapes](https://www.cityscapes-dataset.com/), [BDD-100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/),[Synthia](https://synthia-dataset.net/downloads/) ([SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/download/808/)), [GTAV](https://download.visinf.tu-darmstadt.de/data/from_games/) and [Mapillary Vistas](https://www.mapillary.com/dataset/vistas?pKey=2ix3yvnjy9fwqdzwum3t9g&lat=20&lng=0&z=1.5).

We adopt Class uniform sampling proposed in [this paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf) to handle class imbalance problems. [GTAVUniform](https://github.com/shachoi/RobustNet/blob/0538c69954c030273b3df952f90347572ecac53b/datasets/gtav.py#L306) and [CityscapesUniform](https://github.com/shachoi/RobustNet/blob/0538c69954c030273b3df952f90347572ecac53b/datasets/cityscapes.py#L324) are the datasets to which Class Uniform Sampling is applied.


1. For Cityscapes dataset, download "leftImg8bit_trainvaltest.zip" and "gtFine_trainvaltest.zip" from https://www.cityscapes-dataset.com/downloads/<br>
Unzip the files and make the directory structures as follows.
```
cityscapes
 └ leftImg8bit_trainvaltest
   └ leftImg8bit
     └ train
     └ val
     └ test
 └ gtFine_trainvaltest
   └ gtFine
     └ train
     └ val
     └ test
```
```
bdd-100k
 └ images
   └ train
   └ val
   └ test
 └ labels
   └ train
   └ val
```
```
mapillary
 └ training
   └ images
   └ labels
 └ validation
   └ images
   └ labels
 └ test
   └ images
   └ labels
```

#### We used [GTAV_Split](https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip) to split GTAV dataset into training/validation/test set. Please refer the txt files in in the code of RobustNet [split_data](https://github.com/shachoi/RobustNet/tree/main/split_data).

```
GTAV
 └ images
   └ train
     └ folder
   └ valid
     └ folder
   └ test   
     └ folder
 └ labels
   └ train
     └ folder
   └ valid
     └ folder
   └ test   
     └ folder
```

#### We randomly splitted [Synthia dataset](http://synthia-dataset.net/download/808/) into train/val set. Please refer the txt files in the code of RobustNet [split_data](https://github.com/shachoi/RobustNet/tree/main/split_data).

```
synthia
 └ RGB
   └ train
   └ val
 └ GT
   └ COLOR
     └ train
     └ val
   └ LABELS
     └ train
     └ val
```


step 1 pretrain the self-supervised source domain projection network by run the code python pretrain_autoencoder.py in folder pretrain_ae

If you use DPCL in your research or wish to refer to the baseline results published in our paper, please use the following BibTeX entry.

```BibTeX
@inproceedings{DPCL,
  title={Generalized Semantic Segmentation by Self-Supervised Source Domain Projection and Multi-Level Contrastive Learning},
  author={Liwei Yang, XiangGu, and JianSun},
  journal={AAAI},
  year={2023}
}
```

## Acknowledgement

Code is largely based on RobustNet (https://github.com/shachoi/RobustNet). We use the same way of data splitting with RobustNet. More details can be seen in RobustNet.
