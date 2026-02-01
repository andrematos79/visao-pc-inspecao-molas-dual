from pathlib import Path
import random
import shutil

# ==========================
# CONFIG
# ==========================
SEED = 42
SPLIT = (0.80, 0.10, 0.10)  # train, val, test

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "dataset_mola_roi_raw"          # <-- ajuste se seu nome for outro
OUT_DIR = BASE_DIR / "dataset_mola_roi_split"        # <-- opcional

CLASSES = ["mola_presente", "mola_ausente"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

random.seed(SEED)


def list_images(class_dir: Path):
    if not class_dir.exists():
        return []
    files = []
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def ensure_clean_out():
    # cria estrutura OUT_DIR/train|val|test/<classe>
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            d = OUT_DIR / split / c
            d.mkdir(parents=True, exist_ok=True)


def copy_files(files, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = dst_dir / src.name
        if dst.exists():
            dst = dst_dir / f"{src.stem}__dup{random.randint(1000,9999)}{src.suffix}"
        shutil.copy2(src, dst)


def main():
    print("BASE_DIR:", BASE_DIR)
    print("RAW_DIR:", RAW_DIR)
    print("OUT_DIR:", OUT_DIR)

    if abs(sum(SPLIT) - 1.0) > 1e-6:
        raise ValueError("SPLIT precisa somar 1.0 (ex.: 0.8,0.1,0.1)")

    ensure_clean_out()

    total_all = 0
    for c in CLASSES:
        class_dir = RAW_DIR / c
        imgs = list_images(class_dir)

        print(f"\nClasse '{c}': pasta = {class_dir}")
        print(f"  -> imagens encontradas: {len(imgs)}")

        total_all += len(imgs)

        if len(imgs) == 0:
            if class_dir.exists():
                any_files = [p.name for p in class_dir.iterdir() if p.is_file()]
                print("  Arquivos (não-imagem ou extensão não reconhecida):", any_files[:10])
            continue

        random.shuffle(imgs)

        n = len(imgs)
        n_train = int(n * SPLIT[0])
        n_val = int(n * SPLIT[1])
        n_test = n - n_train - n_val

        train_files = imgs[:n_train]
        val_files = imgs[n_train:n_train + n_val]
        test_files = imgs[n_train + n_val:]

        print(f"  -> split: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

        copy_files(train_files, OUT_DIR / "train" / c)
        copy_files(val_files, OUT_DIR / "val" / c)
        copy_files(test_files, OUT_DIR / "test" / c)

    print("\n✅ Dataset dividido com sucesso!")
    print("Total de imagens (todas classes):", total_all)


if __name__ == "__main__":
    main()
