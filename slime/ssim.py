import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def crop_to_non_black(image):
    """裁剪图像到包含非黑色像素的最小矩形区域"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y + h, x:x + w]
    return image


def pad_and_resize_image_to_size(image, target_size):
    """将图像等比缩放并填充到目标尺寸"""
    height, width = image.shape[:2]
    target_height, target_width = target_size

    # 计算缩放比例
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 等比缩放图像
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 计算填充
    pad_height = target_height - new_height
    pad_width = target_width - new_width

    # 填充图像
    padded_image = cv2.copyMakeBorder(
        resized_image,
        0, pad_height,
        0, pad_width,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # 黑色填充
    )
    return padded_image


def calculate_ssim(image1, image2):
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray_image1, gray_image2, full=True)
    return score


def calculate_psnr(image1, image2):
    return cv2.PSNR(image1, image2)


def calculate_ssim_psnr_with_folder(image_path1, folder_path2):
    image1 = cv2.imread(image_path1)
    if image1 is None:
        raise FileNotFoundError(f"无法读取图像文件：{image_path1}")
    cropped_image1 = crop_to_non_black(image1)
    height1, width1 = cropped_image1.shape[:2]

    results = {}
    for filename in os.listdir(folder_path2):
        file_path = os.path.join(folder_path2, filename)
        if os.path.isfile(file_path) and file_path != image_path1:
            image2 = cv2.imread(file_path)
            if image2 is None:
                print(f"无法读取图像文件：{file_path}")
                continue

            cropped_image2 = crop_to_non_black(image2)

            target_size = (height1, width1)
            padded_image2 = pad_and_resize_image_to_size(cropped_image2, target_size)

            ssim_score = calculate_ssim(cropped_image1, padded_image2)
            psnr_score = calculate_psnr(cropped_image1, padded_image2)

            if ssim_score > 0.6:
                results[filename] = {'SSIM': ssim_score, 'PSNR': psnr_score}

                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(cropped_image1, cv2.COLOR_BGR2RGB))
                plt.title(f'{os.path.basename(image_path1)}')

                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(padded_image2, cv2.COLOR_BGR2RGB))
                plt.title(f'{filename}\nSSIM: {ssim_score:.4f}, PSNR: {psnr_score:.2f} dB')

                # 保存图像而不是显示图像
                output_plot_path = os.path.join("./ssim",
                                                f"comparison_{os.path.basename(image_path1)}_{filename}.png")
                plt.savefig(output_plot_path)
                plt.close()

    return results


# 示例使用
image_path1 = './result/00_1.png'
folder_path2 = './result/'

results = calculate_ssim_psnr_with_folder(image_path1, folder_path2)
for filename, scores in results.items():
    print(f"SSIM and PSNR between {image_path1} and {filename}: SSIM = {scores['SSIM']}, PSNR = {scores['PSNR']}")
