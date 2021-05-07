# PSGMN
code for paper "Pseudo-Siamese Graph Matching Network for Textureless Objects' 6D Pose Estimation". 
If you find this code useful for your research, please consider citing our paper with the following BibTeX entry.
```
@ARTICLE{psgmn,
  author={C. {Wu} and L. {Chen} and Z. {He} and J. {Jiang}},
  journal={IEEE Transactions on Industrial Electronics}, 
  title={Pseudo-Siamese Graph Matching Network for Textureless Objects' 6D Pose Estimation}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIE.2021.3070501}}
``` 

## Installation

1. Set up the python environment:
    ```
    conda create -n psgmn python=3.7
    conda activate psgmn
    ```
    ### install torch 1.5 built for cuda 10.1
    ```
    conda install pytorch==1.5.0 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
    ```
    ### install [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
    ```
    pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-geometric
    ```
    ### install other requirements
    ```
    pip install -r requirements.txt
    ```
    ### compile the cuda extension
    ```
    cd csrc
    python setup.py build_ext --inplace 
    ```
2. Set up datasets:
    Download datasets which are formatted by [PVNet](https://github.com/zju3dv/clean-pvnet):

    (1). [linemod](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EXK2K0B-QrNPi8MYLDFHdB8BQm9cWTxRGV9dQgauczkVYQ?e=beftUz)

    (2). [occlusion linemod](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/ESXrP0zskd5IvvuvG3TXD-4BMgbDrHZ_bevurBrAcKE5Dg?e=r0EgoA)

    Download the simplified mesh models for each object [here](https://ussteducn-my.sharepoint.com/:u:/g/personal/wuchenrui_usst_edu_cn/EaTRLzrbFgxMnJpKLYh2w7ABWtK4-xfrLAmJ9my66uzTKw?e=5sMqkw). Unzip the file and copy it to linemode dataset.

    Make soft links to the datasets.
    ```
    ROOT=/path/to/gsgmn
    cd $ROOT
    mkdir data
    cd data
    ln -s /path/to/linemod linemod
    ln -s /path/to/occlusion_linemod occlusion_linemod
    ```
## Training
Take the training on `ape` as an example.
   run
   ```
   python main_psgmn.py --class_type ape --train True
   ```
## Testing

### Testing on Linemod

We provide the pretrained models of objects on Linemod, which can be found at [here](https://ussteducn-my.sharepoint.com/:f:/g/personal/wuchenrui_usst_edu_cn/EuhOxm1AAOhAh108zNxiZ7UBab41UGRtjX6Z1jw0xQcGEg?e=US6rWq).

Take the testing on `ape` as an example.


1. Download the pretrained model of `ape` and put it to `$ROOT/model/ape/200.pkl`.
2. Test:
    ```
    python main_psgmn.py --class_type ape --eval True
    python main_psgmn.py --class_type ape --occ True --eval True
    ```
