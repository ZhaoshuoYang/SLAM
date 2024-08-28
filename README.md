# SLAM: Segment overlapping objects with SLIME and SAM
![image](https://github.com/user-attachments/assets/252e3d93-bb06-4979-93d9-436792f29126)

This is a project based on SLIME and SAM for single-sample overlap image segmentation. Looking forward to helping you!
# Setup
To begin using SLAM, you first need to create a virtual environment and install the dependencies using the following commands:
```python
conda create -n slam
conda activate slam
pip install -r requirements.txt
```
*** ***For each image and mask pair used for training, validation, or testing with SLAM, their names should match. Furthermore, the images should be in `PNG` format, while the masks should be in `NumPy` format.*** ***
# Start with SAM
Additionally, masks can be generated for images from the command line:
```python
cd sam
python scripts/amg.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input ./toothbrush --output ./output
```
# SLiMe training
First, create a new folder (e.g., `slam/slime/data/train`) and place the training images along with their corresponding masks in that folder (`slam/slime/data/train`). Then, provide the path to the created training data folder (`slam/slime/data/train`) as an argument to `--train_data_dir`. If you have validation data, which will only be used for checkpoint selection, repeat the same process for the validation data (e.g., place the images and masks in `slam/slime/data/val`) and provide the folder's address as an argument to `--val_data_dir`. However, if you don't have validation data, use the address of the training data folder as an argument for `--val_data_dir`.

Next, place the test images in a separate folder (e.g., `slam/slime/data/test`) and specify the path to this folder using `--test_data_dir`. Additionally, you should define a name for the segmented parts within the training images to be used with the `--parts_to_return` argument, including the background. For instance, if you have segmented the brushtooth, you should set `--parts_to_return` to `"background brushtooth"`.

Finally, execute the following command (the main folder obtained after cloning):
```python
cd ../slime
python -m src.main --dataset sample --part_names background toothbrush --train_data_dir ./datasets/toothbrush/train_1 --val_data_dir ./datasets/toothbrush/val --test_data_dir ./datasets/toothbrush/test --train 
```
# Testing with the trained text embeddings
To use the trained text embeddings for testing, run this command:
```python
python -m src.main --dataset sample --checkpoint_dir ./outputs/checkpoints/version_0 --test_data_dir ./dataset/toothbrush/mytest --save_test_predictions
```
![image](https://github.com/user-attachments/assets/b2428632-676d-4851-ac5e-bf540caedd11)
# SLAM post-processing
Next, we will fuse the two results through post-processing operations to obtain the SLAM final result.
```python
python post-processing.py
```
Then, we can use SSIM to calculate the similarity between the segmented object and the target object.
```python
python SSIM.py
```
Finally, we can see the result of the partitioning of the instance.
```python
python show.py
```
![image](https://github.com/user-attachments/assets/d2a41a41-dcc7-400a-8124-012e34b8a187)
# Citation

``` bibtex
@article{yang2024slam,
  title={SLAM: Segment overlapping objects with SLIME and SAM},
  author={Zhaoshuo Yang},
  year={2024}
}
```
