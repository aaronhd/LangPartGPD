## LangPartGPD
1. train:
    ```bash
    python train_SHAPE_grounding.py --model pointnet2_part_seg_ssglang_sent --descrip SHAPE_phy_exp3 --train_mode part-wise --data_mode full --gpu 0
    ```


2. grounding  visualization demo, press Q quit:
    ```bash
    python inference.py 
    ```

3. inference demo:
    ```bash
    python grounding_ros_module_pcl_solo.py  --vis
    ```


