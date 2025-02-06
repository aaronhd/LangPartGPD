## LangPartGPD
1. train:
   512 input points default
    ```bash
    python train.py --mode train --cuda --batch-size 64  --split_mode part-wise  --gpu 0
    ```


3. inference:
    ```bash
    python inference.py 
    ```



