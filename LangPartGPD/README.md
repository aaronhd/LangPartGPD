## LangPartGPD
1. train:
    ```bash
pointnet used  512 input points
python train.py --mode train --cuda --batch-size 64  --split_mode part-wise  --gpu 0
```


2. inference:
    ```bash
    python inference.py 
    ```



