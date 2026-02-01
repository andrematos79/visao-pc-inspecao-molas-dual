# gerar_dataset_roi_dual.py  (VERSÃO CORRIGIDA — BINÁRIO POR ROI)
# Gera ROIs ESQ/DIR a partir de imagens inteiras e salva em:
# OUT_DIR/mola_presente e OUT_DIR/mola_ausente

from pathlib import Path
import cv2

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

# Pasta com as imagens inteiras separadas por classe de origem
# Exemplo de estrutura em IN_DIR:
# dataset_raw/
#   ok/
#   ng_ausente_esq/
#   ng_ausente_dir/
#   ng_ausente_ambas/
IN_DIR  = BASE_DIR / "dataset_raw"

# Saída: dataset "raw" já por ROI e binário (para depois rodar split_dataset.py)
OUT_DIR = BASE_DIR / "dataset_mola_roi_raw"

# Ajuste estes ROIs (em %) com os valores que você está usando no app
# (os mesmos sliders: x0,x1,y0,y1 em %)
ROI = {
    "ESQ": {"x0": 8,  "x1": 35,  "y0": 10, "y1": 82},   # EXEMPLO
    "DIR": {"x0": 74, "x1": 100, "y0": 17, "y1": 83},   # EXEMPLO
}

# Extensões aceitas
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Se você quiser forçar o tamanho da ROI (igual do treino e inferência)
ENABLE_RESIZE = True
IMG_SIZE = (224, 224)  # (W,H) - OpenCV usa (W,H)

# Se True, ignora classes desconhecidas e segue (com WARN).
# Se False, o script para no primeiro erro (mais seguro).
SKIP_UNKNOWN_CLASSES = False


# =========================
# HELPERS
# =========================
def list_images(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.suffix.lower() in EXTS and p.is_file()]

def crop_roi(img, roi_pct):
    h, w = img.shape[:2]
    x0 = int(w * roi_pct["x0"] / 100.0)
    x1 = int(w * roi_pct["x1"] / 100.0)
    y0 = int(h * roi_pct["y0"] / 100.0)
    y1 = int(h * roi_pct["y1"] / 100.0)

    # clamp
    x0 = max(0, min(w - 1, x0))
    x1 = max(1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(1, min(h, y1))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI inválido: {roi_pct} -> px {(x0, x1, y0, y1)}")

    return img[y0:y1, x0:x1].copy()

def maybe_resize(img):
    if not ENABLE_RESIZE:
        return img
    return cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

def save_img(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Falha ao salvar: {path}")

def safe_name(s: str):
    # evita espaços/acentos simples (opcional)
    return s.replace(" ", "_")


# =========================
# MAPEAMENTO DE RÓTULOS (BINÁRIO)
# =========================
# Entrada (foto inteira) -> rótulo por ROI (ESQ/DIR) em 2 classes:
# mola_presente / mola_ausente
def labels_for_rois(src_class: str):
    """
    src_class é o nome da pasta de origem dentro do dataset_raw.
    Retorna dict: {"ESQ": label, "DIR": label}
    """
    s = src_class.lower().strip()

    # OK: as duas molas presentes
    if s == "ok":
        return {"ESQ": "mola_presente", "DIR": "mola_presente"}

    # Ausência por lado
    if s in {"ng_ausente_esq", "ausente_esq"}:
        return {"ESQ": "mola_ausente", "DIR": "mola_presente"}
    if s in {"ng_ausente_dir", "ausente_dir"}:
        return {"ESQ": "mola_presente", "DIR": "mola_ausente"}
    if s in {"ng_ausente_ambas", "ng_ausente_duas", "ausente_ambas", "ausente_duas"}:
        return {"ESQ": "mola_ausente", "DIR": "mola_ausente"}

    # Desalinhamento não entra nesse modelo binário (evita contaminar dataset)
    if "desalinh" in s:
        raise ValueError(
            f"Classe '{src_class}' é desalinhamento. "
            "Neste modelo binário (presente/ausente) isso deve ficar fora do dataset."
        )

    # Classe genérica sem lado definido: erro para não treinar errado
    if s in {"ng_ausente", "ausente"}:
        raise ValueError(
            "Pasta 'ng_ausente' sem lado definido. "
            "Crie subpastas: ng_ausente_esq / ng_ausente_dir / ng_ausente_ambas."
        )

    raise ValueError(f"Classe desconhecida no dataset_raw: '{src_class}'")


# =========================
# MAIN
# =========================
def main():
    print("BASE_DIR:", BASE_DIR)
    print("IN_DIR :", IN_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("ROI ESQ:", ROI["ESQ"])
    print("ROI DIR:", ROI["DIR"])
    print("ENABLE_RESIZE:", ENABLE_RESIZE, "| IMG_SIZE:", IMG_SIZE)

    if not IN_DIR.exists():
        raise FileNotFoundError(f"Não achei {IN_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    total_warn_roi = 0
    total_warn_read = 0
    total_skip_class = 0

    # percorre classes de origem
    class_dirs = [p for p in IN_DIR.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"Nenhuma pasta de classe encontrada em {IN_DIR}")

    for class_dir in sorted(class_dirs):
        src_class = class_dir.name
        imgs = list_images(class_dir)
        print(f"\nClasse origem: {src_class} -> {len(imgs)} imagens")

        if not imgs:
            continue

        try:
            roi_labels = labels_for_rois(src_class)
        except Exception as e:
            msg = f"[WARN] Classe '{src_class}' ignorada: {e}"
            if SKIP_UNKNOWN_CLASSES:
                print(msg)
                total_skip_class += len(imgs)
                continue
            raise

        for img_path in imgs:
            total_in += 1
            img = cv2.imread(str(img_path))
            if img is None:
                total_warn_read += 1
                print("  [WARN] não consegui ler:", img_path)
                continue

            try:
                crop_esq = maybe_resize(crop_roi(img, ROI["ESQ"]))
                crop_dir = maybe_resize(crop_roi(img, ROI["DIR"]))
            except Exception as e:
                total_warn_roi += 1
                print("  [WARN] ROI falhou em:", img_path.name, "|", e)
                continue

            stem = safe_name(img_path.stem)

            # salva com labels corretos por ROI (binário)
            out_esq = OUT_DIR / roi_labels["ESQ"] / f"{stem}__ESQ.jpg"
            out_dir = OUT_DIR / roi_labels["DIR"] / f"{stem}__DIR.jpg"

            save_img(out_esq, crop_esq)
            save_img(out_dir, crop_dir)

            total_out += 2

    print("\n✅ Concluído!")
    print("Total imagens origem        :", total_in)
    print("Total ROIs geradas (ESQ+DIR):", total_out)
    print("WARN leitura (imread)       :", total_warn_read)
    print("WARN ROI inválida           :", total_warn_roi)
    print("Imagens puladas por classe  :", total_skip_class)
    print("\nPróximo passo:")
    print("1) Rode split_dataset.py apontando para OUT_DIR (dataset_mola_roi_raw)")
    print("2) Treine com train_tf_molas.py apontando para o dataset splitado.")
    print("3) Use o app com threshold para MOLA PRESENTE (por ROI) + AND final.")

if __name__ == "__main__":
    main()
