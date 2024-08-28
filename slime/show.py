import cv2
import numpy as np
import os


def overlay_masks_on_image(image_path, mask_paths, output_path, alpha=0.5):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image: {image_path}")
        return

    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # 读取包含颜色信息的掩码
        if mask is None:
            print(f"Error loading mask: {mask_path}")
            continue

        if mask.shape[2] == 4:
            # 分离掩码的颜色和alpha通道
            color_mask = mask[:, :, :3]
            alpha_channel = mask[:, :, 3] / 255.0
        else:
            color_mask = mask
            alpha_channel = np.ones(mask.shape[:2], dtype=np.float32)

        if image.shape[:2] != color_mask.shape[:2]:
            print(f"Resizing mask from {color_mask.shape[:2]} to {image.shape[:2]}")
            color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            alpha_channel = cv2.resize(alpha_channel, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        alpha_mask = alpha * alpha_channel[:, :, np.newaxis]

        image = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0, image)

    cv2.imwrite(output_path, image)
    print(f"Overlay saved to {output_path}")


def process_folders(image_folder, mask_folder, output_folder, alpha=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

    for file in image_files:
        image_path = os.path.join(image_folder, file)
        mask_paths = [os.path.join(mask_folder, mask_file) for mask_file in mask_files if
                      mask_file.startswith(file.split('.')[0])]

        if mask_paths:
            overlay_masks_on_image(image_path, mask_paths, os.path.join(output_folder, file), alpha)


# 使用示例
image_folder = './toothbrush'
mask_folder = './final'
output_folder = './show'

process_folders(image_folder, mask_folder, output_folder)
