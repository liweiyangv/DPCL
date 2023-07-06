# Generalized Semantic Segmentation by Self-Supervised Source Domain Projection and Multi-Level Contrastive Learning (DPCL), AAAI 2023 (Oral Presentation) 
Domain Generalization Semantic Segmentation

<img src="./dpcl_framework.png"></img>

## DPCL - Pytorch

Implementation of <a href="https://doi.org/10.1609/aaai.v37i9.26280">DPCL</a>, domain generalized network for segmentation, in Pytorch. The full paper can be found in <a href="https://arxiv.org/abs/2303.01906">this</a>.

### How to Run DPCL
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
2. You should modify the path in **"<path_to_dpcl>/segmentation_network/config.py"** according to your dataset path.
```
#Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = <YOUR_CITYSCAPES_PATH>
#Mapillary Dataset Dir Location
__C.DATASET.MAPILLARY_DIR = <YOUR_MAPILLARY_PATH>
#GTAV Dataset Dir Location
__C.DATASET.GTAV_DIR = <YOUR_GTAV_PATH>
#BDD-100K Dataset Dir Location
__C.DATASET.BDD_DIR = <YOUR_BDD_PATH>
#Synthia Dataset Dir Location
__C.DATASET.SYNTHIA_DIR = <YOUR_SYNTHIA_PATH>
```
```
3. You can train DPCL with following commands.
```
Step 1 pre-train the source projection network
<path_to_dpcl>/pretrain_ae python pretrain_autoencoder.py
```
```
Step 2 train the segmentation network
<path_to_dpcl>/segmentation_network/$ source ./scripts/train_r50_gtav_dpcl.sh # Train: GTAV, Test: BDD100K, Cityscapes, Mapillary / ResNet50, DPCL
```
### Pretrained Models
You can download the pretrained source projection network evaluated in our paper at [Google Drive](https://drive.google.com/drive/folders/1gEthHAKqhEczRWlonVaOTExszOMyg7ju). Please put the pretrained source projection model in the folder <path_to_dpcl>/pretrain_ae/


step 1 pretrain the self-supervised source domain projection network by run the code python pretrain_autoencoder.py in folder pretrain_ae

If you use DPCL in your research or wish to refer to the baseline results published in our paper, please use the following BibTeX entry.

```BibTeX
@inproceedings{DPCL,
  title={Generalized Semantic Segmentation by Self-Supervised Source Domain Projection and Multi-Level Contrastive Learning},
  author={Liwei Yang, XiangGu, and JianSun},
  journal={AAAI},
  year={2023}
}

## Questions for code
If you meet any questions of our code or paper. Please send email to yangliwei@stu.xjtu.edu.cn at any time. 

## Acknowledgement

Code is largely based on <a href="https://github.com/shachoi/RobustNet">RobustNet</a>. We use the same way of data splitting with RobustNet. More details can be seen in RobustNet.
