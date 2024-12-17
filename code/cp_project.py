import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import os

from cp_hw2 import lRGB2XYZ

def read_image(path, downsample=1, color=True):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(path, flag)
    if downsample > 1:
        image = image[::downsample, ::downsample]
    if color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    else:
        image = image / 255.0
    return image

def save_image(path, image, is_color=False):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if is_color:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def align_images(target, reference, num_correspondences=100, patch_size=50):
    target = target.astype(np.float32)
    reference = reference.astype(np.float32)

    pts_src = []
    pts_dst = []

    for _ in tqdm(range(num_correspondences), desc="Aligning Images"):
        y_ref = np.random.randint(0, reference.shape[0] - patch_size)
        x_ref = np.random.randint(0, reference.shape[1] - patch_size)
        template = reference[y_ref:y_ref+patch_size, x_ref:x_ref+patch_size]

        res = cv2.matchTemplate(target, template, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        x_tar, y_tar = max_loc

        pts_src.append([x_ref, y_ref])
        pts_dst.append([x_tar, y_tar])

    pts_src = np.array(pts_src, dtype=np.float32)
    pts_dst = np.array(pts_dst, dtype=np.float32)

    H, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)

    warped_ref = cv2.warpPerspective(reference, H, (target.shape[1], target.shape[0]))
    return warped_ref

def compute_similarity_map(target_speckle, ref_speckle, patch_size=21):
    half_patch = patch_size // 2
    similarity_map = np.zeros_like(target_speckle, dtype=np.float32)

    for i in tqdm(range(half_patch, target_speckle.shape[0] - half_patch), desc="Computing Similarity Map"):
        for j in range(half_patch, target_speckle.shape[1] - half_patch):
            patch_tar = target_speckle[i - half_patch:i + half_patch + 1, j - half_patch:j + half_patch + 1]
            patch_ref = ref_speckle[i - half_patch:i + half_patch + 1, j - half_patch:j + half_patch + 1]

            mean_tar = np.mean(patch_tar)
            mean_ref = np.mean(patch_ref)
            numerator = np.sum((patch_tar - mean_tar) * (patch_ref - mean_ref))
            denominator = np.sqrt(np.sum((patch_tar - mean_tar)**2) * np.sum((patch_ref - mean_ref)**2)) + 1e-6
            ncc = numerator / denominator
            similarity_map[i, j] = ncc

    similarity_map_normalized = cv2.normalize(similarity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return similarity_map_normalized

def detect_tampering(similarity_map, threshold=50, sigma=3):
    smoothed = gaussian_filter(similarity_map, sigma=sigma)
    _, binary = cv2.threshold(smoothed, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, binary

def visualize_results(original, reference, warped_ref, speckle_tar, speckle_ref, similarity_map, contours):
    warped_ref_color = cv2.cvtColor(warped_ref.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    cv2.drawContours(warped_ref_color, contours, -1, (255, 0, 0), 2)

    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.title('Original Preprocessed')
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Reference Preprocessed')
    plt.imshow(reference, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('Warped Reference')
    plt.imshow(warped_ref, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Similarity Map')
    plt.imshow(similarity_map, cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    N = 1
    data = "data11"
    data_dir = f"data/{data}"
    results_dir = f"results/{data}"
    os.makedirs(results_dir, exist_ok=True)
    target_path = os.path.join(data_dir, "exposure3.tiff")
    reference_path = os.path.join(data_dir, "exposure4.tiff")

    target_rgb = read_image(target_path, downsample=N, color=True)
    reference_rgb = read_image(reference_path, downsample=N, color=True)
    target_lum = lRGB2XYZ(target_rgb)[:, :, 1]
    reference_lum = lRGB2XYZ(reference_rgb)[:, :, 1]
    target_pre = preprocess_image(target_lum)
    reference_pre = preprocess_image(reference_lum)

    warped_ref = align_images(target_pre, reference_pre, num_correspondences=100, patch_size=50)
    similarity_map = compute_similarity_map(target_pre, warped_ref, patch_size=21)
    contours, binary_mask = detect_tampering(similarity_map, threshold=50, sigma=3)

    speckle_tar = cv2.normalize(target_pre, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    speckle_ref = cv2.normalize(warped_ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    visualize_results(target_pre, reference_pre, warped_ref, speckle_tar, speckle_ref, similarity_map, contours)

    save_image(os.path.join(results_dir, f"I_tar_{data}.png"), target_pre, is_color=False)
    save_image(os.path.join(results_dir, f"I_ref_{data}.png"), reference_pre, is_color=False)
    save_image(os.path.join(results_dir, f"I_ref_warped_{data}.png"), warped_ref, is_color=False)
    save_image(os.path.join(results_dir, f"Speckle_tar_{data}.png"), speckle_tar, is_color=False)
    save_image(os.path.join(results_dir, f"Speckle_ref_{data}.png"), speckle_ref, is_color=False)
    save_image(os.path.join(results_dir, f"Similarity_map_{data}.png"), similarity_map, is_color=False)

    warped_ref_color = cv2.cvtColor(warped_ref.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    cv2.drawContours(warped_ref_color, contours, -1, (255, 0, 0), 2)
    save_image(os.path.join(results_dir, f"Tampering_detected_{data}.png"), warped_ref_color, is_color=True)

if __name__ == "__main__":
    main()
