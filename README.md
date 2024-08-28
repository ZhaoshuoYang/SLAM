# SLAM: Segment overlapping objects with SLIME and SAM
# Setup
To begin using SLAM, you first need to create a virtual environment and install the dependencies using the following commands:
```python
conda create -n slam
conda activate slam
pip install -r requirements.txt
```
*** ***For each image and mask pair used for training, validation, or testing with SLAM, their names should match. Furthermore, the images should be in `PNG` format, while the masks should be in `NumPy` format.*** ***
# SLiMe training
First, create a new folder (e.g., `slam/slime/data/train`) and place the training images along with their corresponding masks in that folder (`slam/slime/data/train`). Then, provide the path to the created training data folder (`slam/slime/data/train`) as an argument to `--train_data_dir`. If you have validation data, which will only be used for checkpoint selection, repeat the same process for the validation data (e.g., place the images and masks in `slam/slime/data/val`) and provide the folder's address as an argument to `--val_data_dir`. However, if you don't have validation data, use the address of the training data folder as an argument for `--val_data_dir`.

Next, place the test images in a separate folder (e.g., `slam/slime/data/test`) and specify the path to this folder using `--test_data_dir`. Additionally, you should define a name for the segmented parts within the training images to be used with the `--parts_to_return` argument, including the background. For instance, if you have segmented the brushtooth, you should set `--parts_to_return` to `"background brushtooth"`.

Finally, execute the following command within the slime folder (the main folder obtained after cloning):
```python
cd slime
python -m src.main --dataset sample --part_names background toothbrush --train_data_dir ./datasets/toothbrush/train_1 --val_data_dir ./datasets/toothbrush/val --test_data_dir ./datasets/toothbrush/test --train 
```
# Testing with the trained text embeddings
To use the trained text embeddings for testing, run this command:
```python
python -m src.main --dataset sample --checkpoint_dir ./outputs/checkpoints/version_0 --test_data_dir ./dataset/toothbrush/mytest --save_test_predictions
```
