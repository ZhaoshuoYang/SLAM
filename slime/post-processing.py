import cv2
import glob
import os
import numpy as np
from PIL import Image


def create_overlap_mask(mask1_path, mask2_path):
    # 加载mask图像
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否加载成功
    if mask1 is None:
        print(f"Error loading mask1 image: {mask1_path}")
        return None
    if mask2 is None:
        print(f"Error loading mask2 image: {mask2_path}")
        return None

    # 如果两张mask图像的尺寸不同，将mask1调整为mask2的尺寸
    if mask1.shape != mask2.shape:
        print(f"Resizing mask1 from {mask1.shape} to {mask2.shape}")
        mask1 = cv2.resize(mask1, (mask2.shape[1], mask2.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 找到重叠部分
    overlap_mask = cv2.bitwise_and(mask1, mask2)

    # 计算非零像素数
    non_zero_count = cv2.countNonZero(overlap_mask)

    return overlap_mask, non_zero_count


def process_masks(mask1_path, mask2_folder, output_folder, min_non_zero_count=100, similarity_threshold=0.1, top_n=5):
    # 获取mask2文件夹中的所有图像路径
    mask2_paths = glob.glob(os.path.join(mask2_folder, '*.png'))

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    results = []

    # 对每张mask2图像进行处理
    for mask2_path in mask2_paths:
        print(f"Processing {mask2_path}...")
        result = create_overlap_mask(mask1_path, mask2_path)
        if result is not None:
            overlap_mask, non_zero_count = result
            if non_zero_count >= min_non_zero_count:
                results.append((mask2_path, overlap_mask, non_zero_count))

    # 筛选相对大小相似且面积较大的几个mask
    if results:
        # 根据非零像素数排序
        results.sort(key=lambda x: x[2], reverse=True)

        # 筛选相对大小相似的mask
        filtered_results = []
        for i, (_, _, count) in enumerate(results):
            if i == 0 or abs(count - results[0][2]) / results[0][2] <= similarity_threshold:
                filtered_results.append(results[i])
            if len(filtered_results) >= top_n:
                break

        # 保存筛选后的mask
        for mask2_path, overlap_mask, _ in filtered_results:
            output_filename = f'overlap_{os.path.basename(mask2_path)}'
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, overlap_mask)
            print(f"Saved result to {output_path}")


def image_to_array(image_path):
    """ 将图像转换为二值化的 numpy 数组 """
    image = Image.open(image_path).convert('L')  # 转换为灰度图像
    array = np.array(image)
    return (array > 128).astype(np.uint8)  # 二值化，阈值 128


def is_almost_contained(mask1, mask2, threshold=0.95):
    """ 检查 mask1 是否几乎被 mask2 包含 """
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1)
    if union == 0:
        return False
    return (intersection / union) >= threshold


def is_black_mask(mask):
    """ 检查 mask 是否全黑 """
    return np.sum(mask) == 0


def remove_unwanted_masks(mask_files, threshold=0.95):
    """ 删除全黑和那些几乎被其他 mask 包含的 mask """
    masks = [image_to_array(f) for f in mask_files]
    to_remove = set()
    n = len(masks)

    # 首先标记全黑的 mask
    for i in range(n):
        if is_black_mask(masks[i]):
            to_remove.add(i)

    # 然后标记几乎被其他 mask 包含的 mask
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(n):
            if i != j and j not in to_remove:
                if is_almost_contained(masks[i], masks[j], threshold):
                    to_remove.add(i)
                    break

    return [mask_files[i] for i in range(n) if i not in to_remove]


def deletion(folder_path, output_folder, threshold=0.95):
    """ 主函数，处理文件夹中的 mask 文件 """
    mask_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    filtered_mask_files = remove_unwanted_masks(mask_files, threshold)

    # 保存结果到输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in filtered_mask_files:
        img = Image.open(file)
        img.save(os.path.join(output_folder, os.path.basename(file)))

    print(f"Processed masks saved to {output_folder}")


def generate_colors(num_colors):
    """生成具有大色差的 num_colors 个颜色"""
    colors = []
    for i in range(num_colors):
        hue = (i * 360 / num_colors) % 360  # 色相范围 [0, 360)
        saturation = 255  # 饱和度
        value = 255  # 亮度

        # 将HSV转换为RGB
        color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(color))
    return colors


for i in range(12):
    mask1_path = './outputs/test_results/outputs/lightning_logs/versions_0/' + str(i) + '_eroded_mask.png'
    mask2_path = '../sam/output/' + str(i) + '/'
    finally_path = './finallys/finally_' + str(i)
    # 调用函数
    process_masks(mask1_path, mask2_path, finally_path, min_non_zero_count=2000,
                  similarity_threshold=1.0, top_n=50)

    finally_del_path = './finally_dels/finally_' + str(i) + '_del'  # 输出处理后的 mask 文件夹路径
    threshold = 0.95  # 设置阈值
    deletion(finally_path, finally_del_path, threshold)

    output_path = './final/{:02d}.png'.format(i)

    # 获取所有的mask文件
    mask_files = [f for f in os.listdir(finally_del_path) if f.endswith('.png')]

    # 定义50种颜色
    colors = generate_colors(50)

    # 初始化一个空的合成mask
    combined_mask = None

    # 处理每一个mask文件
    for i, mask_file in enumerate(mask_files):
        mask_path = os.path.join(finally_del_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 检查是否有足够的颜色定义
        if i >= len(colors):
            print(f"Warning: Not enough colors defined for all masks. Mask '{mask_file}' will use default color.")
            color = (255, 255, 255)  # 默认颜色
        else:
            color = colors[i]

        # 转换mask为彩色
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mask[mask > 0] = color

        # 合成mask
        if combined_mask is None:
            combined_mask = color_mask
        else:
            combined_mask = cv2.addWeighted(combined_mask, 1.0, color_mask, 1.0, 0)

    # 将 mask 转换为灰度图像
    gray_mask = cv2.cvtColor(combined_mask, cv2.COLOR_BGR2GRAY)

    # 二值化处理，将非零值设为 255（白色），零值设为 0（黑色）
    _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

    # 查找所有连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # 设置最小面积阈值
    min_area = 500  # 你可以调整这个值

    # 创建新的 mask（保持原始颜色）
    new_mask = np.zeros_like(combined_mask)

    # 保留大于最小面积阈值的区域
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            new_mask[labels == i] = combined_mask[labels == i]

    # 保存合成的mask
    cv2.imwrite(output_path, new_mask)

    print(f"Combined mask saved to {output_path}")
