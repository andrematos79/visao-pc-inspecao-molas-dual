from pathlib import Path
import json
import re
import os
import csv
import io
import time
import random
import shutil
import platform
import subprocess
from datetime import datetime

import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
import textwrap

# Donut (matplotlib) — com fallback se não existir
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


APP_VERSION = "v1.0.0"
APP_STAGE = "Stable"

# ==========================================================
# CONFIG / PATHS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "modelo_molas.keras"
LABELS_PATH = BASE_DIR / "labels.json"
CONFIG_PATH = BASE_DIR / "config_molas.json"
REGISTRY_PATH = BASE_DIR / "models_registry.json"

IMG_SIZE = (224, 224)
DEFAULT_THRESH_PRESENTE = 0.80
DEFAULT_NORMALIZE_LAB = True

ENG_PIN = "1234"  # PIN DO MODO ENGENHARIA

DEFAULT_ROI = {
    "ESQ": {"x0": 8,  "x1": 35,  "y0": 10, "y1": 82},
    "DIR": {"x0": 74, "x1": 100, "y0": 17, "y1": 83},
}

# ==========================================================
# SESSION STATE — INIT (blindado)
# ==========================================================
def init_session():
    # contadores produção
    st.session_state.setdefault("cnt_total", 0)
    st.session_state.setdefault("cnt_ok", 0)
    st.session_state.setdefault("cnt_ng", 0)
    st.session_state.setdefault("cnt_ng_esq", 0)
    st.session_state.setdefault("cnt_ng_dir", 0)

    # histórico p/ charts
    st.session_state.setdefault("history", [])

    # resultados/erros
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("last_result", None)

    # frames
    st.session_state.setdefault("display_frame", None)
    st.session_state.setdefault("last_frame", None)
    st.session_state.setdefault("frozen", False)
    st.session_state.setdefault("frozen_frame", None)

    # câmera
    st.session_state.setdefault("cap", None)
    st.session_state.setdefault("camera_on", False)
    st.session_state.setdefault("cam_index_last", 0)

    # modelo/labels
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("labels", None)

    # modo + PIN
    st.session_state.setdefault("user_mode", "OPERADOR")
    st.session_state.setdefault("eng_unlocked", False)

    # modelo selecionado
    st.session_state.setdefault("selected_model_key", "MODELO_PADRAO")
    st.session_state.setdefault("product_model", st.session_state.get("selected_model_key", "MODELO_PADRAO"))

    # turnos
    st.session_state.setdefault("shift", 1)

    # aprendizado
    st.session_state.setdefault("learning_last_saved", None)

init_session()

# ==========================================================
# CONFIG POR MODELO (setup de linha) — ROI/Threshold por produto
# ==========================================================
CONFIG_DEFAULT_PATH = (BASE_DIR / "config_molas.json")
CONFIGS_DIR = (BASE_DIR / "configs")
CONFIGS_DIR.mkdir(exist_ok=True)

def _safe_model_key(model_key: str) -> str:
    s = str(model_key).strip()
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s or "MODEL_DEFAULT"

def model_config_path(model_key: str) -> Path:
    return CONFIGS_DIR / f"{_safe_model_key(model_key)}.json"

def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def get_effective_config(model_key: str) -> dict:
    """
    Carrega config específica do MODELO, se existir.
    Caso contrário, usa o config_molas.json (default/global).
    """
    p = model_config_path(model_key)
    cfg = load_json(p)
    if cfg:
        return cfg
    return load_json(CONFIG_DEFAULT_PATH)

def apply_config_to_session(cfg: dict) -> None:
    """
    Aplica config nas chaves padrão do session_state.
    Essas chaves serão usadas pelos sliders (única fonte).
    """
    if not cfg:
        return

    st.session_state["threshold_presente"] = float(cfg.get("threshold_presente", 0.40))
    st.session_state["normalize_lab_equalize"] = bool(cfg.get("normalize_lab_equalize", True))

    roi = cfg.get("roi", {}) or {}
    esq = roi.get("ESQ", {}) or {}
    dirr = roi.get("DIR", {}) or {}

    st.session_state["roi_esq_x0"] = int(esq.get("x0", 8))
    st.session_state["roi_esq_x1"] = int(esq.get("x1", 35))
    st.session_state["roi_esq_y0"] = int(esq.get("y0", 10))
    st.session_state["roi_esq_y1"] = int(esq.get("y1", 82))

    st.session_state["roi_dir_x0"] = int(dirr.get("x0", 74))
    st.session_state["roi_dir_x1"] = int(dirr.get("x1", 100))
    st.session_state["roi_dir_y0"] = int(dirr.get("y0", 17))
    st.session_state["roi_dir_y1"] = int(dirr.get("y1", 83))

def collect_config_from_session() -> dict:
    """
    Gera payload para salvar no JSON do modelo atual.
    """
    return {
        "threshold_presente": float(st.session_state.get("threshold_presente", 0.40)),
        "normalize_lab_equalize": bool(st.session_state.get("normalize_lab_equalize", True)),
        "roi": {
            "ESQ": {
                "x0": int(st.session_state.get("roi_esq_x0", 8)),
                "x1": int(st.session_state.get("roi_esq_x1", 35)),
                "y0": int(st.session_state.get("roi_esq_y0", 10)),
                "y1": int(st.session_state.get("roi_esq_y1", 82)),
            },
            "DIR": {
                "x0": int(st.session_state.get("roi_dir_x0", 74)),
                "x1": int(st.session_state.get("roi_dir_x1", 100)),
                "y0": int(st.session_state.get("roi_dir_y0", 17)),
                "y1": int(st.session_state.get("roi_dir_y1", 83)),
            },
        },
    }

# garantir defaults coerentes (1ª execução)
if "threshold_presente" not in st.session_state:
    apply_config_to_session(get_effective_config(st.session_state.get("selected_model_key", "MODELO_PADRAO")))

# ==========================================================
# DATASET (APRENDIZADO) — CAPTURA E SALVAMENTO
# ==========================================================
DATASET_ROOT = BASE_DIR / "dataset_products"
DATASET_ROOT.mkdir(exist_ok=True)

def safe_slug(s: str) -> str:
    keep = []
    for ch in str(s).strip():
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    out = "".join(keep).strip("_")
    return out if out else "PRODUTO"

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def ensure_product_dirs(prod_key: str) -> dict:
    prod = safe_slug(prod_key)
    base = DATASET_ROOT / prod
    raw_ok = base / "raw" / "ok"
    raw_ng = base / "raw" / "ng"

    roi_esq_ok = base / "roi" / "ESQ" / "mola_presente"
    roi_esq_ng = base / "roi" / "ESQ" / "mola_ausente"
    roi_dir_ok = base / "roi" / "DIR" / "mola_presente"
    roi_dir_ng = base / "roi" / "DIR" / "mola_ausente"

    for p in [raw_ok, raw_ng, roi_esq_ok, roi_esq_ng, roi_dir_ok, roi_dir_ng]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "base": base,
        "raw_ok": raw_ok, "raw_ng": raw_ng,
        "roi_esq_ok": roi_esq_ok, "roi_esq_ng": roi_esq_ng,
        "roi_dir_ok": roi_dir_ok, "roi_dir_ng": roi_dir_ng,
    }

def count_jpgs(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.jpg")))

def save_jpg(path: Path, img_bgr: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

def build_sample_names(prod_key: str, label_simple: str, cam_idx: int) -> tuple[str, str, str]:
    ts = now_stamp()
    prod = safe_slug(prod_key)
    raw_name = f"{ts}__PROD-{prod}__RAW__{label_simple}__cam{cam_idx}.jpg"
    esq_name = f"{ts}__PROD-{prod}__ROI-ESQ__{('mola_presente' if label_simple=='OK' else 'mola_ausente')}.jpg"
    dir_name = f"{ts}__PROD-{prod}__ROI-DIR__{('mola_presente' if label_simple=='OK' else 'mola_ausente')}.jpg"
    return raw_name, esq_name, dir_name

def read_one_frame(cap: cv2.VideoCapture):
    if cap is None or not cap.isOpened():
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame

def read_fresh_frame(
    cap: cv2.VideoCapture,
    flush_grabs: int = 6,
    sleep_ms: int = 15,
    extra_reads: int = 0,
):
    if cap is None or not cap.isOpened():
        return None

    for _ in range(int(flush_grabs)):
        cap.grab()
        if sleep_ms > 0:
            time.sleep(float(sleep_ms) / 1000.0)

    ok, frame = cap.read()
    if not ok or frame is None:
        return None

    for _ in range(int(extra_reads)):
        ok2, frame2 = cap.read()
        if ok2 and frame2 is not None:
            frame = frame2

    return frame

def capture_source_frame_for_learning() -> np.ndarray | None:
    if st.session_state.get("camera_on") and st.session_state.get("cap") is not None:
        src = read_fresh_frame(
            st.session_state["cap"],
            flush_grabs=10,
            sleep_ms=10,
            extra_reads=2
        )
        if src is not None:
            st.session_state["last_frame"] = src.copy()
        return src

    lf = st.session_state.get("last_frame")
    return lf.copy() if lf is not None else None

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def crop_roi_percent(frame_bgr: np.ndarray, x0p, x1p, y0p, y1p) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x0 = int(clamp01(x0p / 100.0) * w)
    x1 = int(clamp01(x1p / 100.0) * w)
    y0 = int(clamp01(y0p / 100.0) * h)
    y1 = int(clamp01(y1p / 100.0) * h)

    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])

    if x1 - x0 < 10 or y1 - y0 < 10:
        return frame_bgr.copy()

    return frame_bgr[y0:y1, x0:x1].copy()

def equalize_lab_bgr(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.equalizeHist(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def save_learning_sample(
    label_simple: str,
    mode_capture: str,
    save_raw: bool = True,
    jpeg_quality: int = 92,
) -> dict:
    prod_key = st.session_state.get("selected_model_key", "MODELO_PADRAO")
    dirs = ensure_product_dirs(prod_key)

    src = capture_source_frame_for_learning()
    if src is None:
        raise RuntimeError("Sem frame para salvar (ligue a câmera ou tenha um last_frame).")

    roi_esq = crop_roi_percent(
        src,
        int(st.session_state.get("roi_esq_x0", DEFAULT_ROI["ESQ"]["x0"])),
        int(st.session_state.get("roi_esq_x1", DEFAULT_ROI["ESQ"]["x1"])),
        int(st.session_state.get("roi_esq_y0", DEFAULT_ROI["ESQ"]["y0"])),
        int(st.session_state.get("roi_esq_y1", DEFAULT_ROI["ESQ"]["y1"])),
    )

    roi_dir = crop_roi_percent(
        src,
        int(st.session_state.get("roi_dir_x0", DEFAULT_ROI["DIR"]["x0"])),
        int(st.session_state.get("roi_dir_x1", DEFAULT_ROI["DIR"]["x1"])),
        int(st.session_state.get("roi_dir_y0", DEFAULT_ROI["DIR"]["y0"])),
        int(st.session_state.get("roi_dir_y1", DEFAULT_ROI["DIR"]["y1"])),
    )

    if bool(st.session_state.get("normalize_lab_equalize", DEFAULT_NORMALIZE_LAB)):
        roi_esq = equalize_lab_bgr(roi_esq)
        roi_dir = equalize_lab_bgr(roi_dir)

    raw_name, esq_name, dir_name = build_sample_names(
        prod_key,
        label_simple,
        int(st.session_state.get("cam_index_last", 0)),
    )

    saved = {"product": prod_key, "label": label_simple, "mode": mode_capture, "raw": None, "esq": None, "dir": None}

    if save_raw:
        raw_dir = dirs["raw_ok"] if label_simple == "OK" else dirs["raw_ng"]
        raw_path = raw_dir / raw_name
        save_jpg(raw_path, src, quality=jpeg_quality)
        saved["raw"] = str(raw_path)

    if mode_capture in ("DUAL", "ESQ"):
        esq_dir = dirs["roi_esq_ok"] if label_simple == "OK" else dirs["roi_esq_ng"]
        esq_path = esq_dir / esq_name
        save_jpg(esq_path, roi_esq, quality=jpeg_quality)
        saved["esq"] = str(esq_path)

    if mode_capture in ("DUAL", "DIR"):
        dir_dir = dirs["roi_dir_ok"] if label_simple == "OK" else dirs["roi_dir_ng"]
        dir_path = dir_dir / dir_name
        save_jpg(dir_path, roi_dir, quality=jpeg_quality)
        saved["dir"] = str(dir_path)

    st.session_state["learning_last_saved"] = saved
    return saved

def learning_counts(prod_key: str) -> dict:
    dirs = ensure_product_dirs(prod_key)
    return {
        "raw_ok": count_jpgs(Path(dirs["raw_ok"])),
        "raw_ng": count_jpgs(Path(dirs["raw_ng"])),
        "esq_ok": count_jpgs(Path(dirs["roi_esq_ok"])),
        "esq_ng": count_jpgs(Path(dirs["roi_esq_ng"])),
        "dir_ok": count_jpgs(Path(dirs["roi_dir_ok"])),
        "dir_ng": count_jpgs(Path(dirs["roi_dir_ng"])),
        "base": str(dirs["base"]),
    }

# ==========================================================
# DATASET (APRENDIZADO) — SPLIT train/val/test
# ==========================================================
def list_jpgs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.jpg"))

def safe_rmtree(path: Path) -> None:
    if path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)

def ensure_split_dirs(base: Path) -> dict:
    out = {}
    for side in ["ESQ", "DIR"]:
        for split in ["train", "val", "test"]:
            for cls in ["mola_presente", "mola_ausente"]:
                p = base / "roi_split" / side / split / cls
                p.mkdir(parents=True, exist_ok=True)
                out[(side, split, cls)] = p
    return out

def split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float):
    s = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if s <= 0:
        raise ValueError("Ratios inválidos (soma <= 0).")
    tr = float(train_ratio) / s
    vr = float(val_ratio) / s
    ti = int(round(n * tr))
    vi = int(round(n * vr))
    if ti < 0: ti = 0
    if vi < 0: vi = 0
    if ti + vi > n:
        vi = max(0, n - ti)
    te = n - (ti + vi)
    return ti, vi, te

def make_split_for_side(
    product_base: Path,
    side: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
    overwrite: bool = True,
) -> dict:
    side = side.upper().strip()
    if side not in ("ESQ", "DIR"):
        raise ValueError("side deve ser ESQ ou DIR.")

    src_ok = product_base / "roi" / side / "mola_presente"
    src_ng = product_base / "roi" / side / "mola_ausente"

    ok_files = list_jpgs(src_ok)
    ng_files = list_jpgs(src_ng)

    if len(ok_files) == 0 or len(ng_files) == 0:
        raise RuntimeError(f"Sem imagens suficientes em roi/{side}. OK={len(ok_files)} NG={len(ng_files)}")

    split_root = product_base / "roi_split" / side
    if overwrite:
        safe_rmtree(split_root)

    split_dirs = ensure_split_dirs(product_base)
    rng = random.Random(int(seed))

    def do_one_class(files: list[Path], cls: str) -> dict:
        files = files.copy()
        rng.shuffle(files)

        n = len(files)
        n_tr, n_va, n_te = split_indices(n, train_ratio, val_ratio, test_ratio)

        tr_files = files[:n_tr]
        va_files = files[n_tr:n_tr+n_va]
        te_files = files[n_tr+n_va:]

        for f in tr_files:
            shutil.copy2(str(f), str(split_dirs[(side, "train", cls)] / f.name))
        for f in va_files:
            shutil.copy2(str(f), str(split_dirs[(side, "val", cls)] / f.name))
        for f in te_files:
            shutil.copy2(str(f), str(split_dirs[(side, "test", cls)] / f.name))

        return {"total": n, "train": len(tr_files), "val": len(va_files), "test": len(te_files)}

    out_ok = do_one_class(ok_files, "mola_presente")
    out_ng = do_one_class(ng_files, "mola_ausente")

    return {"side": side, "ok": out_ok, "ng": out_ng, "base": str(product_base)}

def make_split_product(
    prod_key: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
    overwrite: bool = True,
) -> dict:
    dirs = ensure_product_dirs(prod_key)
    base = Path(dirs["base"])

    res_esq = make_split_for_side(base, "ESQ", train_ratio, val_ratio, test_ratio, seed, overwrite)
    res_dir = make_split_for_side(base, "DIR", train_ratio, val_ratio, test_ratio, seed, overwrite)

    return {"product": prod_key, "ESQ": res_esq, "DIR": res_dir, "split_root": str(base / "roi_split")}

# ==========================================================
# REGISTRY (CADASTRO DE MODELOS)
# ==========================================================
def registry_fallback() -> dict:
    return {
        "MODELO_PADRAO": {
            "descricao": "Baseline v1.0.0 - Mola DUAL",
            "ativo": True,
            "model_path": str(MODEL_PATH.name),
            "labels_path": str(LABELS_PATH.name),
            "config_path": str(CONFIG_PATH.name),
        }
    }

def load_registry(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and len(data) > 0:
                return data
        except Exception:
            pass
    return registry_fallback()

def save_registry(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def ensure_registry_file() -> None:
    if not REGISTRY_PATH.exists():
        save_registry(REGISTRY_PATH, registry_fallback())

def get_active_models(registry: dict) -> dict:
    return {k: v for k, v in registry.items() if isinstance(v, dict) and v.get("ativo", False)}

def resolve_model_paths(entry: dict) -> tuple[Path, Path, Path]:
    mp = BASE_DIR / str(entry.get("model_path", MODEL_PATH.name))
    lp = BASE_DIR / str(entry.get("labels_path", LABELS_PATH.name))
    cp = BASE_DIR / str(entry.get("config_path", CONFIG_PATH.name))
    return mp, lp, cp

# ==========================================================
# MODEL HELPERS
# ==========================================================
def load_labels(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"labels não encontrado: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("labels.json deve ser uma LISTA, ex: ['mola_ausente','mola_presente']")
    return data

@st.cache_resource(show_spinner=False)
def load_model_cached(path_str: str) -> tf.keras.Model:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"modelo não encontrado: {path}")
    return tf.keras.models.load_model(path, compile=False)

def preprocess_bgr_for_model(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    return np.expand_dims(img, axis=0)

def predict_one(model: tf.keras.Model, labels: list[str], frame_bgr: np.ndarray):
    x = preprocess_bgr_for_model(frame_bgr)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    cls = labels[idx]
    conf = float(probs[idx])
    return cls, conf, probs

def prob_of_class(labels: list[str], probs: np.ndarray, class_name: str) -> float:
    if class_name not in labels:
        return 0.0
    return float(probs[labels.index(class_name)])

def safe_release_cap():
    cap = st.session_state.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state["cap"] = None
    st.session_state["camera_on"] = False

def ensure_model_loaded_or_raise():
    if st.session_state.get("model") is not None and st.session_state.get("labels") is not None:
        return

    paths = st.session_state.get("selected_model_paths")
    if paths and len(paths) == 3:
        model_p = Path(paths[0])
        labels_p = Path(paths[1])
    else:
        model_p = MODEL_PATH
        labels_p = LABELS_PATH

    st.session_state["labels"] = load_labels(labels_p)
    st.session_state["model"] = load_model_cached(str(model_p))

# ==========================================================
# LOG CSV
# ==========================================================
def append_log_csv(row: dict):
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"inspecao_molas_{today}.csv"

    fieldnames = [
        "timestamp", "modelo", "turno", "resultado_final",
        "cs_code", "cs_detail", "p_esq", "p_dir", "th_presente",
        "camera_index", "directshow", "source",
        "total", "ok", "ng", "yield_pct",
        "test_time_sec", "start_time", "end_time",
    ]

    file_exists = log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

def get_cs_code(res: dict, th: float) -> tuple[str, str]:
    ok_esq = res.get("ok_esq", False)
    ok_dir = res.get("ok_dir", False)

    if ok_esq and ok_dir:
        return "OK", "OK"
    if (not ok_esq) and ok_dir:
        return "NG_ESQ", f"p_esq<{th:.2f}"
    if ok_esq and (not ok_dir):
        return "NG_DIR", f"p_dir<{th:.2f}"
    return "NG_AMBAS", f"p_esq<{th:.2f} | p_dir<{th:.2f}"

# ==========================================================
# STREAMLIT APP
# ==========================================================
st.set_page_config(page_title="Inspeção de Molas — DUAL (modo estável)", layout="wide")
st.title("Inspeção de Molas — DUAL (modo estável) ✅")

# ==========================================================
# CSS
# ==========================================================
st.markdown("""<style>
.app-footer {
    position: fixed; bottom: 6px; right: 12px;
    font-size: 11px; color: #9ca3af; opacity: 0.85;
    z-index: 9999; pointer-events: none;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
.roi-box { background-color: #f4f6f8; border: 1px solid #d0d4d9; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
.roi-title { font-weight: 600; font-size: 16px; margin-bottom: 4px; }
.roi-caption { font-size: 12px; color: #6b7280; margin-bottom: 10px; }
.roi-frame { border-radius: 6px; padding: 6px; }
.roi-ok { border: 2px solid #22c55e; }
.roi-ng { border: 2px solid #dc2626; }
.roi-bar {
    height: 44px; border-radius: 6px; margin: 8px 0 12px 0;
    border: 1px solid #d0d4d9; display: flex; align-items: center; justify-content: center;
    font-size: 35px; font-weight: 700; letter-spacing: 1px; color: #ffffff;
    text-transform: uppercase; line-height: 1;
}
.roi-bar-ok { background: #22c55e; }
.roi-bar-ng { background: #ef4444; }
.result-box { border-radius: 10px; padding: 16px; margin-top: 10px; margin-bottom: 14px; text-align: center; }
.result-ok { background-color: #dcfce7; border: 2px solid #22c55e; color: #166534; }
.result-ng { background-color: #fee2e2; border: 2px solid #dc2626; color: #7f1d1d; }
.result-text { font-size: 42px; font-weight: 800; letter-spacing: 1px; }
.result-details { font-size: 14px; margin-top: 8px; }
.kpi-grid{ display:grid; grid-template-columns:repeat(3, minmax(120px, 1fr)); gap:8px; max-width: 580px; width: 100%; }
.kpi-card{ background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:6px 8px; min-height:52px; }
.kpi-label { font-size:12px; color:#6b7280; margin-bottom:0px; line-height: 1.05; }
.kpi-value { font-size:22px; font-weight:800; color:#111827; margin:2px 0 0 0; line-height:1.0; }
.kpi-wide{ grid-column:1/-1; display:flex; justify-content:space-between; align-items:baseline; min-height:44px; padding:8px 12px; margin-bottom: 2px; }
.kpi-value-yield { font-size:22px; font-weight:900; line-height: 1.0; }
.compact-divider{ height: 10px; background-color: #eef2f6; border: 1px solid #d0d4d9; border-radius: 999px; margin: 8px 0 10px 0; width: 100%; box-sizing: border-box; }
.resumo-card{ background:#ffffff; border:1px solid #d0d4d9; border-radius:10px; padding:10px 10px 8px 10px; margin-top:0px; }
.resumo-title{ font-weight:700; font-size:20px; margin:0 0 6px 0; }
.pie-wrap{ margin-top:-60px; display:flex; justify-content:center; align-items:flex-start; }
section[data-testid="stSidebar"] { padding-top: 6px; padding-bottom: 6px; }
section[data-testid="stSidebar"] label { margin-bottom: 1px !important; font-size: 11px; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] select { min-height: 32px !important; padding-top: 4px !important; padding-bottom: 4px !important; font-size: 14px; }
section[data-testid="stSidebar"] button { min-height: 32px !important; padding-top: 2px !important; padding-bottom: 2px !important; font-size: 12px; }
section[data-testid="stSidebar"] hr { margin-top: 6px; margin-bottom: 6px; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { margin: 6px 0 4px 0 !important; font-size: 16px !important; }
</style>""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR — LOGO + SOBRE
# ==========================================================
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo_empresa.jpg"

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.caption("⚠️ Logo não encontrado em assets/logo_empresa.jpg")

    st.markdown("---")

    with st.expander("ℹ️ Sobre o Sistema", expanded=False):
        st.markdown("### Inspeção de Molas — DUAL")
        st.markdown(f"- **Versão:** {APP_VERSION}")
        st.markdown(f"- **Status:** {APP_STAGE}")
        st.markdown(f"- **Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        st.markdown("- **Empresa:** Salcomp")
        st.markdown("- **Engenheiro Responsável:** André Gama de Matos")
        st.markdown("---")
        st.markdown("**Ambiente de Execução**")
        st.markdown(f"- **Sistema Operacional:** {platform.system()} {platform.release()}")
        st.markdown(f"- **Python:** {platform.python_version()}")
        st.markdown(f"- **OpenCV:** {cv2.__version__}")
        st.markdown(f"- **TensorFlow:** {tf.__version__}")

# ==========================================================
# SIDEBAR — MODO + PIN
# ==========================================================
st.sidebar.header("Modo")
c1, c2 = st.sidebar.columns(2)

with c1:
    if st.sidebar.button("👷 Operador", use_container_width=True):
        st.session_state["user_mode"] = "OPERADOR"
        st.session_state["eng_unlocked"] = False
        st.rerun()

with c2:
    if st.sidebar.button("🛠 Eng.", use_container_width=True):
        st.session_state["user_mode"] = "ENG"
        st.rerun()

st.sidebar.caption(f"Modo atual: **{st.session_state.get('user_mode','OPERADOR')}**")

if st.session_state.get("user_mode") == "ENG" and not st.session_state.get("eng_unlocked", False):
    st.sidebar.warning("Digite o PIN para acessar o modo Eng.")
    pin = st.sidebar.text_input("PIN", type="password")
    if pin == ENG_PIN:
        st.session_state["eng_unlocked"] = True
        st.sidebar.success("Liberado ✅")
        st.rerun()
    elif pin != "":
        st.sidebar.error("PIN incorreto ❌")

is_eng = (st.session_state.get("user_mode") == "ENG" and st.session_state.get("eng_unlocked", False))

# ==========================================================
# SIDEBAR — PRODUÇÃO (MODELO/LINHA)
# ==========================================================
ensure_registry_file()
registry = load_registry(REGISTRY_PATH)
ativos = get_active_models(registry)

st.sidebar.subheader("Produção")

if is_eng:
    options_models = list(registry.keys()) if registry else ["MODELO_PADRAO"]
    caption_models = "Modelo (linha) — Engenharia"
else:
    options_models = list(ativos.keys()) if ativos else ["MODELO_PADRAO"]
    caption_models = "Modelo (linha) — Operador (somente seleção)"

def fmt_model(k: str) -> str:
    d = registry.get(k, {})
    desc = d.get("descricao", "")
    tag = "" if d.get("ativo", False) else " [INATIVO]"
    return f"{k} — {desc}{tag}" if desc else f"{k}{tag}"

current_key = st.session_state.get("selected_model_key", "MODELO_PADRAO")
if current_key not in options_models:
    current_key = options_models[0] if options_models else "MODELO_PADRAO"

selected_key = st.sidebar.selectbox(
    caption_models,
    options=options_models,
    index=options_models.index(current_key) if current_key in options_models else 0,
    format_func=fmt_model
)

# Troca de modelo: segura e SEM autoload (evita tela branca)
if selected_key != st.session_state.get("selected_model_key"):
    entry = registry.get(selected_key, registry_fallback()["MODELO_PADRAO"])
    mp, lp, cp = resolve_model_paths(entry)

    st.session_state["selected_model_key"] = selected_key
    st.session_state["selected_model_desc"] = str(entry.get("descricao", ""))
    st.session_state["selected_model_paths"] = (str(mp), str(lp), str(cp))

    st.session_state["model"] = None
    st.session_state["labels"] = None

    # recarrega config do modelo selecionado (por arquivo do modelo)
    try:
        cfg_model = load_json(Path(cp))
        if not cfg_model:
            cfg_model = get_effective_config(selected_key)
    except Exception:
        cfg_model = get_effective_config(selected_key)

    apply_config_to_session(cfg_model)

    # limpa estados visuais
    st.session_state["frozen"] = False
    st.session_state["frozen_frame"] = None
    st.session_state["last_frame"] = None
    st.session_state["last_result"] = None
    st.session_state["last_error"] = None

    safe_release_cap()
    st.sidebar.info("🔄 Modelo trocado. Ligue a câmera novamente e faça a próxima inspeção.")
    st.rerun()

st.session_state["product_model"] = st.session_state.get("selected_model_key", "MODELO_PADRAO")

st.sidebar.selectbox(
    "Turno",
    options=[1, 2, 3],
    index=[1, 2, 3].index(int(st.session_state.get("shift", 1)))
          if int(st.session_state.get("shift", 1)) in [1, 2, 3] else 0,
    key="shift"
)

now_str = datetime.now().strftime("%d/%m/%y %H:%M:%S")
st.sidebar.markdown("""
<style>
.time-label { font-size: 12px; color: #6b7280; margin-bottom: 2px; }
.time-box {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 8px 10px; color: #374151; font-size: 13px; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)
st.sidebar.markdown('<div class="time-label">Time</div>', unsafe_allow_html=True)
st.sidebar.markdown(f'<div class="time-box">{now_str}</div>', unsafe_allow_html=True)

st.sidebar.divider()

# ==========================================================
# BOTÕES CONFIG DO MODELO — SOMENTE ENG LIBERADO
# ==========================================================
if is_eng:
    st.sidebar.subheader("Config do Modelo (Eng.)")

    if st.sidebar.button("💾 Salvar config deste modelo"):
        mk = st.session_state.get("selected_model_key", "MODELO_PADRAO")
        p = model_config_path(mk)
        payload = collect_config_from_session()
        save_json(p, payload)
        st.sidebar.success(f"Salvo: {p.name}")

    if st.sidebar.button("↩️ Recarregar config deste modelo"):
        mk = st.session_state.get("selected_model_key", "MODELO_PADRAO")
        cfg = get_effective_config(mk)
        apply_config_to_session(cfg)
        st.sidebar.info("Config recarregada do arquivo.")
        st.rerun()

st.sidebar.divider()

# ==========================================================
# SIDEBAR — CÂMERA
# ==========================================================
st.sidebar.header("Câmera")
cam_index = st.sidebar.number_input("Índice da câmera (0,1,2...)", min_value=0, max_value=10, value=0, step=1)
use_dshow = st.sidebar.checkbox("Usar DirectShow (Windows)", value=True)

col_cam_btns = st.sidebar.columns(2)
with col_cam_btns[0]:
    btn_cam_on = st.sidebar.button("📷 Ligar", use_container_width=True)
with col_cam_btns[1]:
    btn_cam_off = st.sidebar.button("⛔ Desligar", use_container_width=True)

btn_capture = st.sidebar.button("📸 Capturar + Inferir (DUAL)", type="primary", use_container_width=True)
btn_live = st.sidebar.button("▶️ LIVE", use_container_width=True)

st.session_state["cam_index_last"] = int(cam_index)

# ==========================================================
# SIDEBAR — APRENDIZADO (ENG)
# ==========================================================
if is_eng:
    st.sidebar.divider()
    st.sidebar.header("📚 Aprendizado (Eng.)")

    prod_key = st.session_state.get("selected_model_key", "MODELO_PADRAO")
    st.sidebar.caption(f"Produto atual: **{prod_key}**")

    mode_capture = st.sidebar.radio(
        "Modo de captura",
        options=["DUAL", "ESQ", "DIR"],
        index=0,
        horizontal=True,
        help="DUAL salva ESQ+DIR. ESQ salva apenas ESQ. DIR salva apenas DIR."
    )

    save_raw = st.sidebar.checkbox("Salvar também RAW (frame inteiro)", value=True)
    jpeg_q = st.sidebar.slider("Qualidade JPG", 70, 98, 92, 1)

    c_ok, c_ng = st.sidebar.columns(2)
    with c_ok:
        btn_save_ok = st.sidebar.button("✅ Salvar OK", use_container_width=True)
    with c_ng:
        btn_save_ng = st.sidebar.button("❌ Salvar NG", use_container_width=True)

    try:
        cnt = learning_counts(prod_key)
        st.sidebar.markdown("**Contagem (produto atual):**")
        st.sidebar.write(f"RAW OK: {cnt['raw_ok']} | RAW NG: {cnt['raw_ng']}")
        st.sidebar.write(f"ESQ OK: {cnt['esq_ok']} | ESQ NG: {cnt['esq_ng']}")
        st.sidebar.write(f"DIR OK: {cnt['dir_ok']} | DIR NG: {cnt['dir_ng']}")
        st.sidebar.caption(f"Base: `{cnt['base']}`")
    except Exception:
        st.sidebar.warning("Dataset ainda vazio ou não inicializado.")
        cnt = None

    if btn_save_ok or btn_save_ng:
        try:
            label_simple = "OK" if btn_save_ok else "NG"
            save_learning_sample(
                label_simple=label_simple,
                mode_capture=mode_capture,
                save_raw=save_raw,
                jpeg_quality=int(jpeg_q),
            )
            st.sidebar.success(f"Amostra salva ({label_simple}) ✅")
        except Exception as e:
            st.sidebar.error(f"Falha ao salvar amostra: {e}")

    with st.sidebar.expander("📦 Preparar dataset (Split train/val/test)", expanded=False):
        st.caption("Gera cópia das imagens em roi_split/ESQ e roi_split/DIR (train/val/test).")

        c_r1, c_r2, c_r3 = st.columns(3)
        with c_r1:
            train_ratio = st.number_input("Train", min_value=0.10, max_value=0.95, value=0.70, step=0.05)
        with c_r2:
            val_ratio = st.number_input("Val", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
        with c_r3:
            test_ratio = st.number_input("Test", min_value=0.05, max_value=0.50, value=0.10, step=0.05)

        seed = st.number_input("Seed (reprodutível)", min_value=0, max_value=999999, value=42, step=1)
        overwrite = st.checkbox("Sobrescrever split existente", value=True)

        btn_make_split = st.button("🚀 Gerar Split agora", use_container_width=True)

        if btn_make_split:
            try:
                cnt_local = learning_counts(prod_key)
                min_ok = min(cnt_local["esq_ok"], cnt_local["dir_ok"])
                min_ng = min(cnt_local["esq_ng"], cnt_local["dir_ng"])

                if min_ok < 10 or min_ng < 10:
                    st.warning(f"Poucas imagens para split. Sugestão: >=10 por classe/lado. "
                               f"Min OK={min_ok}, Min NG={min_ng}")

                result = make_split_product(
                    prod_key=prod_key,
                    train_ratio=float(train_ratio),
                    val_ratio=float(val_ratio),
                    test_ratio=float(test_ratio),
                    seed=int(seed),
                    overwrite=bool(overwrite),
                )

                st.success("Split gerado com sucesso ✅")
                st.write(f"Destino: `{result['split_root']}`")

                st.markdown("**Resumo ESQ**")
                st.write("OK:", result["ESQ"]["ok"])
                st.write("NG:", result["ESQ"]["ng"])

                st.markdown("**Resumo DIR**")
                st.write("OK:", result["DIR"]["ok"])
                st.write("NG:", result["DIR"]["ng"])

            except Exception as e:
                st.error(f"Falha ao gerar split: {e}")

# ==========================================================
# SIDEBAR — CONFIG (apenas ENG liberado)
# ==========================================================
show_debug = False
if is_eng:
    st.sidebar.divider()
    st.sidebar.header("Config (Eng.)")

    st.slider("Threshold mínimo p/ MOLA PRESENTE", 0.0, 1.0,
              value=float(st.session_state.get("threshold_presente", 0.40)),
              step=0.01, key="threshold_presente")

    st.checkbox("Normalizar ROI (LAB equalize)",
                value=bool(st.session_state.get("normalize_lab_equalize", True)),
                key="normalize_lab_equalize")

    st.sidebar.subheader("ROI (%)")
    # (se quiser sliders de ROI aqui, você pode adicionar depois)

    show_debug = st.sidebar.checkbox("Mostrar debug", value=False)

# ==========================================================
# Camera ON/OFF
# ==========================================================
if btn_cam_on:
    safe_release_cap()
    st.session_state["display_frame"] = None

    backend = cv2.CAP_DSHOW if use_dshow else cv2.CAP_ANY
    cap = cv2.VideoCapture(int(cam_index), backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        st.session_state["camera_on"] = False
        st.session_state["last_error"] = "Não consegui abrir a câmera. Tente outro índice."
        st.session_state["cap"] = None
    else:
        st.session_state["cap"] = cap
        st.session_state["camera_on"] = True
        st.session_state["last_error"] = None
        st.session_state["frozen"] = False
        st.session_state["frozen_frame"] = None

if btn_cam_off:
    safe_release_cap()
    st.session_state["display_frame"] = None
    st.session_state["frozen"] = False
    st.session_state["frozen_frame"] = None

if btn_live:
    st.session_state["display_frame"] = None
    st.session_state["frozen"] = False
    st.session_state["frozen_frame"] = None

# ==========================================================
# Infer DUAL
# ==========================================================
def infer_dual_on_frame(frame_bgr: np.ndarray):
    if st.session_state.get("model") is None or st.session_state.get("labels") is None:
        raise RuntimeError("Modelo não carregado.")

    esq_x0 = int(st.session_state.get("roi_esq_x0", DEFAULT_ROI["ESQ"]["x0"]))
    esq_x1 = int(st.session_state.get("roi_esq_x1", DEFAULT_ROI["ESQ"]["x1"]))
    esq_y0 = int(st.session_state.get("roi_esq_y0", DEFAULT_ROI["ESQ"]["y0"]))
    esq_y1 = int(st.session_state.get("roi_esq_y1", DEFAULT_ROI["ESQ"]["y1"]))

    dir_x0 = int(st.session_state.get("roi_dir_x0", DEFAULT_ROI["DIR"]["x0"]))
    dir_x1 = int(st.session_state.get("roi_dir_x1", DEFAULT_ROI["DIR"]["x1"]))
    dir_y0 = int(st.session_state.get("roi_dir_y0", DEFAULT_ROI["DIR"]["y0"]))
    dir_y1 = int(st.session_state.get("roi_dir_y1", DEFAULT_ROI["DIR"]["y1"]))

    th_presente = float(st.session_state.get("threshold_presente", DEFAULT_THRESH_PRESENTE))
    normalize_roi = bool(st.session_state.get("normalize_lab_equalize", DEFAULT_NORMALIZE_LAB))

    roi_esq = crop_roi_percent(frame_bgr, esq_x0, esq_x1, esq_y0, esq_y1)
    roi_dir = crop_roi_percent(frame_bgr, dir_x0, dir_x1, dir_y0, dir_y1)

    if normalize_roi:
        roi_esq = equalize_lab_bgr(roi_esq)
        roi_dir = equalize_lab_bgr(roi_dir)

    cls_esq, conf_esq, probs_esq = predict_one(st.session_state["model"], st.session_state["labels"], roi_esq)
    cls_dir, conf_dir, probs_dir = predict_one(st.session_state["model"], st.session_state["labels"], roi_dir)

    p_pres_esq = prob_of_class(st.session_state["labels"], probs_esq, "mola_presente")
    p_pres_dir = prob_of_class(st.session_state["labels"], probs_dir, "mola_presente")

    ok_esq = (p_pres_esq >= th_presente)
    ok_dir = (p_pres_dir >= th_presente)
    aprovado = ok_esq and ok_dir

    return {
        "roi_esq": roi_esq,
        "roi_dir": roi_dir,
        "cls_esq": cls_esq,
        "conf_esq": conf_esq,
        "p_pres_esq": p_pres_esq,
        "ok_esq": ok_esq,
        "cls_dir": cls_dir,
        "conf_dir": conf_dir,
        "p_pres_dir": p_pres_dir,
        "ok_dir": ok_dir,
        "aprovado": aprovado,
    }

def update_metrics_and_history(res: dict):
    st.session_state["cnt_total"] += 1

    if res.get("aprovado", False):
        st.session_state["cnt_ok"] += 1
    else:
        st.session_state["cnt_ng"] += 1
        if not res.get("ok_esq", True):
            st.session_state["cnt_ng_esq"] += 1
        if not res.get("ok_dir", True):
            st.session_state["cnt_ng_dir"] += 1

    st.session_state["history"].append({
        "n": int(st.session_state.get("cnt_total", 0)),
        "aprovado": int(bool(res.get("aprovado", False))),
        "ok_esq": int(bool(res.get("ok_esq", False))),
        "ok_dir": int(bool(res.get("ok_dir", False))),
        "p_esq": float(res.get("p_pres_esq", 0.0)),
        "p_dir": float(res.get("p_pres_dir", 0.0)),
        "ng_esq": int(not bool(res.get("ok_esq", True))),
        "ng_dir": int(not bool(res.get("ok_dir", True))),
    })

# ==========================================================
# ACTIONS — Capturar + Inferir
# ==========================================================
if btn_capture:
    st.session_state["last_error"] = None

    try:
        ensure_model_loaded_or_raise()
    except Exception as e:
        st.session_state["last_error"] = f"Falha ao carregar modelo atual: {e}"
        st.session_state["last_result"] = None
    else:
        src = None
        if st.session_state.get("camera_on") and st.session_state.get("cap") is not None:
            src = read_fresh_frame(
                st.session_state["cap"],
                flush_grabs=12,
                sleep_ms=10,
                extra_reads=2
            )
            if src is not None:
                st.session_state["last_frame"] = src.copy()
        else:
            lf = st.session_state.get("last_frame")
            src = lf.copy() if lf is not None else None

        if src is None:
            st.session_state["last_error"] = "Sem imagem para inferir (ligue a câmera e capture)."
            st.session_state["last_result"] = None
        else:
            st.session_state["display_frame"] = src.copy()
            st.session_state["last_frame"] = src.copy()

            start_dt = datetime.now()
            try:
                res = infer_dual_on_frame(src)
                end_dt = datetime.now()
                test_time_sec = (end_dt - start_dt).total_seconds()

                st.session_state["last_result"] = res
                update_metrics_and_history(res)

                timestamp = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                th_local = float(st.session_state.get("threshold_presente", DEFAULT_THRESH_PRESENTE))
                cs_code, cs_detail = get_cs_code(res, th_local)

                total = int(st.session_state.get("cnt_total", 0))
                ok = int(st.session_state.get("cnt_ok", 0))
                ng = int(st.session_state.get("cnt_ng", 0))
                yield_pct = round((ok / total * 100.0), 2) if total > 0 else 0.0

                row = {
                    "timestamp": timestamp,
                    "modelo": st.session_state.get("product_model", ""),
                    "turno": st.session_state.get("shift", ""),
                    "resultado_final": "OK" if res.get("aprovado", False) else "NG",
                    "cs_code": cs_code,
                    "cs_detail": cs_detail,
                    "p_esq": round(float(res.get("p_pres_esq", 0.0)), 4),
                    "p_dir": round(float(res.get("p_pres_dir", 0.0)), 4),
                    "th_presente": float(th_local),
                    "camera_index": int(cam_index),
                    "directshow": bool(use_dshow),
                    "source": "camera",
                    "total": total,
                    "ok": ok,
                    "ng": ng,
                    "yield_pct": yield_pct,
                    "test_time_sec": f"{test_time_sec:.3f}",
                    "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                }

                append_log_csv(row)

            except Exception as e:
                st.session_state["last_error"] = f"Erro na inferência: {e}"
                st.session_state["last_result"] = None

# ==========================================================
# Frame live / display_frame
# ==========================================================
frame = None
if st.session_state.get("display_frame") is not None:
    frame = st.session_state["display_frame"].copy()
elif st.session_state.get("camera_on") and st.session_state.get("cap") is not None:
    frame = read_one_frame(st.session_state["cap"])
    if frame is not None:
        st.session_state["last_frame"] = frame.copy()
else:
    lf = st.session_state.get("last_frame")
    frame = lf.copy() if lf is not None else None

# ==========================================================
# MAIN — mensagens de erro
# ==========================================================
if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])

# ==========================================================
# ✅ RESUMO (PRODUÇÃO)
# ==========================================================
total = int(st.session_state.get("cnt_total", 0))
ok = int(st.session_state.get("cnt_ok", 0))
ng = int(st.session_state.get("cnt_ng", 0))
yield_pct = (ok / total * 100.0) if total > 0 else 0.0

kpi_html = textwrap.dedent(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Total</div>
    <div class="kpi-value">{total}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">OK</div>
    <div class="kpi-value">{ok}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">NG</div>
    <div class="kpi-value">{ng}</div>
  </div>
  <div class="kpi-card kpi-wide">
    <div class="kpi-label">Yield (%)</div>
    <div class="kpi-value kpi-value-yield">{yield_pct:.2f}</div>
  </div>
</div>
""")

# ==========================================================
# LAYOUT
# ==========================================================
colA, colB = st.columns([2.0, 1.3], gap="medium")

with colA:
    with st.container(border=True):
        st.markdown("#### Visualização")
        if frame is None:
            st.warning("Sem frame (ligue a câmera).")
        else:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=800)

    st.markdown('<div class="compact-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="resumo-card">', unsafe_allow_html=True)
    st.markdown('<div class="resumo-title">Resumo (Produção)</div>', unsafe_allow_html=True)

    c_kpi, c_pie = st.columns([3.2, 1.0], gap="small", vertical_alignment="top")

    with c_kpi:
        st.markdown(kpi_html, unsafe_allow_html=True)

    with c_pie:
        st.markdown('<div class="pie-wrap">', unsafe_allow_html=True)

        if total > 0:
            if HAS_MPL:
                fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=110)
                ax.pie(
                    [ok, ng],
                    startangle=90,
                    counterclock=False,
                    wedgeprops=dict(width=0.35),
                    labels=None
                )
                ax.text(0, 0.15, "Yield", ha="center", va="center", fontsize=10, fontweight="bold")
                ax.text(0, -0.15, f"{yield_pct:.2f}%", ha="center", va="center", fontsize=10)
                ax.set_aspect("equal")
                ax.set_axis_off()
                st.pyplot(fig, use_container_width=False)
            else:
                st.warning("matplotlib não instalado — sem donut.")
        else:
            st.caption("Sem dados para o gráfico (Total = 0).")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    try:
        res = st.session_state.get("last_result")

        if res is not None:
            cls_esq = "roi-ok" if res.get("ok_esq", False) else "roi-ng"
            cls_dir = "roi-ok" if res.get("ok_dir", False) else "roi-ng"
            bar_esq = "roi-bar-ok" if res.get("ok_esq", False) else "roi-bar-ng"
            bar_dir = "roi-bar-ok" if res.get("ok_dir", False) else "roi-bar-ng"

            st.markdown('<div class="roi-box">', unsafe_allow_html=True)
            st.markdown('<div class="roi-title">ROIs das Molas</div>', unsafe_allow_html=True)
            st.markdown('<div class="roi-caption">Recortes usados na inferência (ESQ e DIR).</div>', unsafe_allow_html=True)

            c_esq, c_dir = st.columns(2, gap="small")

            with c_esq:
                st.markdown(f'<div class="roi-bar {bar_esq}">{"OK" if res.get("ok_esq", False) else "NG"}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="roi-frame {cls_esq}">', unsafe_allow_html=True)
                st.markdown("**Mola ESQ (ROI)**")
                roi_img = res.get("roi_esq", None)
                if roi_img is not None:
                    st.image(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB), width=190)
                else:
                    st.caption("ROI ESQ indisponível.")
                st.markdown("</div>", unsafe_allow_html=True)

            with c_dir:
                st.markdown(f'<div class="roi-bar {bar_dir}">{"OK" if res.get("ok_dir", False) else "NG"}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="roi-frame {cls_dir}">', unsafe_allow_html=True)
                st.markdown("**Mola DIR (ROI)**")
                roi_img = res.get("roi_dir", None)
                if roi_img is not None:
                    st.image(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB), width=190)
                else:
                    st.caption("ROI DIR indisponível.")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Ainda não foi feita uma inspeção (sem ROIs para mostrar).")

        if res is not None:
            aprovado = res.get("aprovado", False)
            cls_result = "result-ok" if aprovado else "result-ng"
            txt_result = "✅ APROVADO" if aprovado else "❌ REPROVADO"

            st.markdown(
                f"""
                <div class="result-box {cls_result}">
                    <div class="result-text">{txt_result}</div>
                    <div class="result-details">
                        ESQ: p(mola_presente) = {res.get('p_pres_esq', 0.0):.3f} → {'OK' if res.get('ok_esq', False) else 'NG'}<br>
                        DIR: p(mola_presente) = {res.get('p_pres_dir', 0.0):.3f} → {'OK' if res.get('ok_dir', False) else 'NG'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if len(st.session_state.get("history", [])) > 1:
            with st.expander("📈 Qualidade (Gráficos)", expanded=False):
                import pandas as pd
                df = pd.DataFrame(st.session_state["history"])
                df["ok_cum"] = df["aprovado"].cumsum()
                df["yield_cum"] = 100.0 * df["ok_cum"] / df["n"]

                st.caption("Tendência de Yield (%)")
                st.line_chart(df.set_index("n")[["yield_cum"]])

                st.caption("Defeitos por lado (NG)")
                defects = {
                    "ESQ": int(st.session_state.get("cnt_ng_esq", 0)),
                    "DIR": int(st.session_state.get("cnt_ng_dir", 0))
                }
                st.bar_chart(defects)

    except Exception as e:
        st.error(f"Erro ao renderizar painel direito: {e}")
        if show_debug:
            st.exception(e)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown(
    f"""
    <div class="app-footer">
        {APP_VERSION} · {APP_STAGE} · Developed by André Gama de Matos · Software Engineer
    </div>
    """,
    unsafe_allow_html=True
)

if show_debug:
    st.write("DEBUG:")
    st.write("selected_model_key:", st.session_state.get("selected_model_key"))
    st.write("selected_model_paths:", st.session_state.get("selected_model_paths"))
    st.write("camera_on:", st.session_state.get("camera_on"))
    st.write("last_error:", st.session_state.get("last_error"))
