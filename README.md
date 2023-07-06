# DPCL
Domain Generalization Semantic Segmentation

<img src="./dpcl_framework.png"></img>


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

Code is largely based on RobustNet (https://github.com/shachoi/RobustNet).
