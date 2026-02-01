import cv2
import numpy as np
import os
from tqdm import tqdm
import glob

np.random.seed(42)

BASE_DIR = "dataset"
CLASS_DIRS = ["ok", "ng_ausente", "ng_desalinhada"]
OUTPUT_PER_CLASS = 500

os.makedirs("dataset_aug", exist_ok=True)

def rotate(img):
    ang = np.random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), ang, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def noise(img):
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def brightness(img):
    alpha = np.random.uniform(0.8, 1.2)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def augment(img):
    ops = [rotate, noise, brightness]
    for op in ops:
        img = op(img)
    return img

print("\n📸 GERANDO DATASET AUMENTADO...\n")

for cls in CLASS_DIRS:
    print(f"Classe: {cls}")

    files = glob.glob(f"{BASE_DIR}/{cls}/*")
    if len(files) == 0:
        print(f"❌ ERRO: sem imagens base em {cls}")
        continue

    base_img_path = files[0]   # pega qualquer arquivo da pasta
    base_img = cv2.imread(base_img_path)

    if base_img is None:
        print(f"❌ ERRO: não consegui ler {base_img_path}")
        continue

    out_dir = f"dataset_aug/{cls}"
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(OUTPUT_PER_CLASS)):
        aug_img = augment(base_img)
        cv2.imwrite(f"{out_dir}/{cls}_{i:04d}.jpg", aug_img)

print("\n✅ Dataset aumentado gerado com sucesso!")
