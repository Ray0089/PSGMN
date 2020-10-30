# PSGMN
code for paper "Pseudo-Siamese Graph Matching Network for Texture-less Objects' 6D Pose Estimation"

Right now, the code only consists of the evaluation part. We are collating the training part of the code. It will be available when we finish the clean-up.
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
2. Set up datasets:
    ```
    ROOT=/path/to/gsgmn
    cd $ROOT/data
    ln -s /path/to/linemod linemod
    ln -s /path/to/occlusion_linemod occlusion_linemod

Download datasets which are formatted by [PVNet](https://github.com/zju3dv/clean-pvnet):
1. [linemod](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EXK2K0B-QrNPi8MYLDFHdB8BQm9cWTxRGV9dQgauczkVYQ?e=beftUz)

2. [occlusion linemod](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/ESXrP0zskd5IvvuvG3TXD-4BMgbDrHZ_bevurBrAcKE5Dg?e=r0EgoA)

Download the simplified mesh models for each object [here](https://ussteducn-my.sharepoint.com/:u:/g/personal/wuchenrui_usst_edu_cn/EaTRLzrbFgxMnJpKLYh2w7ABWtK4-xfrLAmJ9my66uzTKw?e=5sMqkw). Unzip the file and copy it to linemode dataset.

## Testing

### Testing on Linemod

We provide the pretrained models of objects on Linemod, which can be found at [here](https://ussteducn-my.sharepoint.com/:f:/g/personal/wuchenrui_usst_edu_cn/EuhOxm1AAOhAh108zNxiZ7UBab41UGRtjX6Z1jw0xQcGEg?e=US6rWq).

Take the testing on `cat` as an example.


1. Download the pretrained model of `cat` and put it to `$ROOT/model/cat/200.pth`.
2. Test:
    ```
    python main_psgmn.py --class_type cat
    python main_psgmn.py --class_type cat --occ True
    ```
