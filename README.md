# LangPartGPD

```bash
conda create --name langshape python=3.8
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# test on 4.8.2
pip install mayavi  
pip install tqdm
pip install scipy==1.10.1
pip install h5py==3.9.0
pip install PyQt5 
```

## LangPartGPD

1. Clone this repository:
    
    ```bash
    cd $HOME/code
    git clone https:/xxx.git
    
    ```
    
2. Install our modified dex-net (Modify from [Berkeley Automation Lab: dex-net](https://github.com/BerkeleyAutomation/dex-net))
    
    ```bash
    cd $HOME/code/PointNetGPD/dex-net
    python setup.py develop
    
    ```
    
3. LangSHAPE dataset and pretrained model are available from [project website](https://sites.google.com/view/lang-shape/dataset)

## 3D part language grounding

1. LangSHAPE dataset and pretrained model are available from [project website](https://sites.google.com/view/lang-shape/dataset)

## Acknowledgment

- [PointNetGPD](https://github.com/lianghongzhuo/PointNetGPD)
- [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
