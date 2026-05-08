
# ==========================================================
# SVC MOLAS - CORE INDUSTRIAL V1.6
# Arquitetura estável do core terminal + inferência industrial v18
# Sem Streamlit controlando sensor/câmera/inferência.
# ==========================================================

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

try:
    import serial
except Exception:
    serial = None


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "modelo_molas.keras"
LABELS_PATH = BASE_DIR / "labels.json"
CONFIG_PATH = BASE_DIR / "config_molas.json"
PROD_MODEL_DIR = BASE_DIR / "models" / "mobilenetv2_prod"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

STATUS_DIR = BASE_DIR / "runtime_status"
STATUS_DIR.mkdir(exist_ok=True)
LAST_RESULT_JSON = STATUS_DIR / "last_result.json"
LAST_RESULT_TMP = STATUS_DIR / "last_result.tmp"
SUMMARY_JSON = STATUS_DIR / "summary.json"
SUMMARY_TMP = STATUS_DIR / "summary.tmp"
HEARTBEAT_JSON = STATUS_DIR / "heartbeat.json"
HEARTBEAT_TMP = STATUS_DIR / "heartbeat.tmp"

CAM_INDEX = 0
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200

SETTLE_S = 0.30
REARM_ZERO_REQUIRED = True
# v8.1-fastline-fix: após sensor voltar para 0, aguarda e limpa buffer serial
# para evitar que um PRESENT=1 antigo dispare foto da base vazia.
REARM_CLEAR_BUFFER_S = 0.25
# Durante o settle, monitora se o sensor voltou para 0. Se voltar, cancela a foto.
SETTLE_CANCEL_ON_ZERO = True
MAX_CYCLES = 0  # 0 = infinito

IMG_SIZE = (224, 224)
DEFAULT_THRESH_PRESENTE = 0.50
DEFAULT_THR_NG_OK = 0.45
DEFAULT_THR_NG_NG = 0.60
# v1.6.5: ESQ mantém bypass para desalinhamento; DIR fica mais conservador.
# Motivo: v1.6.4 ficou estável e acertou mola ausente, mas gerou falsos NG_MISALIGNED no DIR.
DEFAULT_THR_NG_NG_ESQ = 0.9995
DEFAULT_THR_NG_OK_ESQ = 0.9900
DEFAULT_THR_NG_NG_DIR = 0.85
DEFAULT_THR_NG_OK_DIR = 0.60
DEFAULT_NORMALIZE_LAB = True
DEFAULT_TEMPORAL_N_FRAMES = 1
DEFAULT_TEMPORAL_DELAY_MS = 25

DEFAULT_ROI = {
    "ESQ": {"x0": 8,  "x1": 35,  "y0": 10, "y1": 82},
    "DIR": {"x0": 74, "x1": 100, "y0": 17, "y1": 83},
}


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{ts} | {msg}", flush=True)


def atomic_write_json(path_tmp: Path, path_final: Path, data: dict):
    path_tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    path_tmp.replace(path_final)


def write_heartbeat(status="running", cycle=None):
    try:
        atomic_write_json(HEARTBEAT_TMP, HEARTBEAT_JSON, {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "cycle": cycle,
            "core_version": "v19-final-v8.1.2-stable-reset",
        })
    except Exception:
        pass


def update_summary(final_result: str):
    summary = {"total": 0, "ok": 0, "ng": 0, "last_update": None}
    if SUMMARY_JSON.exists():
        try:
            summary.update(json.loads(SUMMARY_JSON.read_text(encoding="utf-8")))
        except Exception:
            pass

    summary["total"] = int(summary.get("total", 0)) + 1
    if final_result == "OK":
        summary["ok"] = int(summary.get("ok", 0)) + 1
    else:
        summary["ng"] = int(summary.get("ng", 0)) + 1

    total = max(1, int(summary["total"]))
    summary["yield_percent"] = round((int(summary["ok"]) / total) * 100.0, 2)
    summary["last_result"] = final_result
    summary["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_json(SUMMARY_TMP, SUMMARY_JSON, summary)
    return summary


def parse_serial_line(line: str):
    if not line:
        return None
    s = line.strip().replace("\r", "").replace("\n", "")
    low = s.lower()
    if low in ("1", "0"):
        return int(low)
    m = re.search(r'\b(present|sensor|p)\s*[:=]\s*([01])\b', low)
    if m:
        return int(m.group(2))
    bits = re.findall(r'\b[01]\b', low)
    if bits:
        return int(bits[-1])
    return None


def load_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def crop_roi_percent(frame_bgr: np.ndarray, x0p, x1p, y0p, y1p) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x0 = int(clamp01(float(x0p) / 100.0) * w)
    x1 = int(clamp01(float(x1p) / 100.0) * w)
    y0 = int(clamp01(float(y0p) / 100.0) * h)
    y1 = int(clamp01(float(y1p) / 100.0) * h)
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


def preprocess_bgr_for_model(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    return np.expand_dims(img, axis=0)


def load_labels(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("labels.json inválido")
    return data


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


def safe_class_index(class_names, target_name, fallback_idx=0):
    """Evita erro de interpretação quando o production_package salva pos_idx diferente.
    Para o modelo de desalinhamento, sempre queremos a probabilidade da classe NG_MISALIGNED.
    """
    if class_names and target_name in class_names:
        return int(class_names.index(target_name))
    return int(fallback_idx)


def load_production_package(outputs_dir: str):
    pkg_path = os.path.join(outputs_dir, "production_package.json")
    if not os.path.isfile(pkg_path):
        return None
    with open(pkg_path, "r", encoding="utf-8") as f:
        pkg = json.load(f)
    class_names = pkg["class_names"]
    pos_name = pkg["pos_class_name"]
    pos_idx = int(pkg["pos_class_index"])
    thr = float(pkg["best_threshold_ng"]["thr"])
    img_size = tuple(pkg.get("img_size", [224, 224]))
    return class_names, pos_name, pos_idx, thr, img_size


def load_mobilenetv2_prod_model(outputs_dir: str):
    model_path = os.path.join(outputs_dir, "model_final.keras")
    if not os.path.isfile(model_path):
        alt = os.path.join(outputs_dir, "best_model.keras")
        if os.path.isfile(alt):
            model_path = alt
        else:
            return None, None
    return tf.keras.models.load_model(model_path, compile=False), model_path


def infer_mobilenetv2_prod(bgr_img, model, class_names, pos_idx, thr_ng, img_size=(224,224)):
    if model is None or class_names is None:
        return "OK", 0.0, {"OK": 1.0}
    rgb = bgr_img[..., ::-1]
    x = tf.image.resize(rgb, img_size, method="bilinear")
    x = tf.cast(x, tf.float32)
    x = tf.expand_dims(x, axis=0)
    p = model.predict(x, verbose=0)[0]
    p = np.asarray(p, dtype=np.float32)
    probs = {class_names[i]: float(p[i]) for i in range(len(class_names))}

    # FIX v1.6.1: não confiar cegamente no pos_idx salvo no pacote.
    # O modelo de produção tem classes=['NG_MISALIGNED', 'OK']; logo p_ng deve vir da classe NG_MISALIGNED.
    idx_ng = safe_class_index(class_names, "NG_MISALIGNED", fallback_idx=pos_idx)
    prob_ng = float(p[idx_ng])
    pred_label = "NG_MISALIGNED" if prob_ng >= thr_ng else "OK"
    return pred_label, prob_ng, probs


def decide_misaligned_status(prob_ng, prob_ok, thr_ng_ok, thr_ng_ng, margin_abs=0.10):
    prob_ng = float(prob_ng)
    prob_ok = float(prob_ok)
    margin = prob_ng - prob_ok
    if prob_ng >= thr_ng_ng and margin >= margin_abs:
        return "NG_MISALIGNED", "NG_STRONG", margin, False
    if prob_ng >= thr_ng_ng and margin < margin_abs:
        return "OK", "ATTENTION", margin, True
    if prob_ng >= thr_ng_ok:
        return "OK", "ATTENTION", margin, True
    return "OK", "OK_SAFE", margin, False


class IndustrialModels:
    def __init__(self):
        log(f"Carregando legacy model: {MODEL_PATH}")
        self.legacy_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        self.labels = load_labels(LABELS_PATH)
        log(f"Labels: {self.labels}")

        self.prod_model = None
        self.prod_class_names = None
        self.prod_pos_idx = 0
        self.prod_img_size = (224, 224)
        self.prod_thr_ng = DEFAULT_THR_NG_NG

        pkg = load_production_package(str(PROD_MODEL_DIR))
        if pkg is not None:
            class_names, pos_name, pos_idx, thr, img_size = pkg
            model, model_path = load_mobilenetv2_prod_model(str(PROD_MODEL_DIR))
            if model is not None:
                self.prod_model = model
                self.prod_class_names = class_names
                # FIX v1.6.1: usar índice da classe pelo nome, não apenas o pos_idx do pacote.
                self.prod_pos_idx = safe_class_index(class_names, "NG_MISALIGNED", fallback_idx=pos_idx)
                self.prod_img_size = tuple(img_size)
                self.prod_thr_ng = float(thr)
                idx_ok = safe_class_index(class_names, "OK", fallback_idx=1)
                log(f"MobileNetV2 produção carregada: {model_path} | classes={class_names} | pkg_pos_idx={pos_idx} | idx_ng={self.prod_pos_idx} | idx_ok={idx_ok}")


def get_roi_cfg(cfg, side):
    # aceita config com chaves diretas ou usa DEFAULT_ROI.
    r = None
    try:
        r = cfg.get("roi", {}).get(side)
    except Exception:
        r = None
    if not r:
        r = DEFAULT_ROI[side]
    return r


def infer_dual_on_frame_v18(frame_bgr, models: IndustrialModels, cfg: dict):
    r_esq = get_roi_cfg(cfg, "ESQ")
    r_dir = get_roi_cfg(cfg, "DIR")

    roi_esq = crop_roi_percent(frame_bgr, r_esq["x0"], r_esq["x1"], r_esq["y0"], r_esq["y1"])
    roi_dir = crop_roi_percent(frame_bgr, r_dir["x0"], r_dir["x1"], r_dir["y0"], r_dir["y1"])

    normalize_roi = bool(cfg.get("normalize_lab_equalize", DEFAULT_NORMALIZE_LAB))
    if normalize_roi:
        roi_esq_inf = equalize_lab_bgr(roi_esq)
        roi_dir_inf = equalize_lab_bgr(roi_dir)
    else:
        roi_esq_inf = roi_esq
        roi_dir_inf = roi_dir

    # v1.6.3: teste cirurgico. A mola ESQ e espelhada em relacao a DIR.
    # Para o classificador de desalinhamento APENAS, espelhamos a ROI ESQ.
    # A deteccao de mola ausente permanece intacta.
    left_flip_for_misaligned = bool(cfg.get("left_flip_for_misaligned", True))
    roi_esq_mis = cv2.flip(roi_esq_inf, 1) if left_flip_for_misaligned else roi_esq_inf
    roi_dir_mis = roi_dir_inf

    th_presente = float(cfg.get("threshold_presente", DEFAULT_THRESH_PRESENTE))
    thr_ng_ok = float(cfg.get("threshold_ng_ok", DEFAULT_THR_NG_OK))
    thr_ng_ng = float(cfg.get("threshold_ng_ng", DEFAULT_THR_NG_NG))

    # v1.6.2: thresholds separados por lado.
    # O log mostrou p_ng_esq saturando em ~0.97-0.996 mesmo com produto OK,
    # então o lado ESQ fica conservador para evitar falsa rejeição em massa.
    thr_ng_ok_esq = float(cfg.get("threshold_ng_ok_esq", DEFAULT_THR_NG_OK_ESQ))
    thr_ng_ng_esq = float(cfg.get("threshold_ng_ng_esq", DEFAULT_THR_NG_NG_ESQ))
    thr_ng_ok_dir = float(cfg.get("threshold_ng_ok_dir", DEFAULT_THR_NG_OK_DIR))
    thr_ng_ng_dir = float(cfg.get("threshold_ng_ng_dir", DEFAULT_THR_NG_NG_DIR))

    margin_abs = float(cfg.get("prod_margin_abs", 0.10))

    cls_pres_esq, conf_esq, probs_pres_esq = predict_one(models.legacy_model, models.labels, roi_esq_inf)
    cls_pres_dir, conf_dir, probs_pres_dir = predict_one(models.legacy_model, models.labels, roi_dir_inf)

    p_pres_esq = prob_of_class(models.labels, probs_pres_esq, "mola_presente")
    p_pres_dir = prob_of_class(models.labels, probs_pres_dir, "mola_presente")

    missing_esq = (p_pres_esq < th_presente)
    missing_dir = (p_pres_dir < th_presente)

    cls_mis_esq = "OK"
    cls_mis_dir = "OK"
    prob_ng_esq = 0.0
    prob_ng_dir = 0.0
    prob_ok_esq = 1.0
    prob_ok_dir = 1.0
    decision_band_esq = "MISSING" if missing_esq else "OK_SAFE"
    decision_band_dir = "MISSING" if missing_dir else "OK_SAFE"
    margin_esq = 1.0
    margin_dir = 1.0
    attention_esq = False
    attention_dir = False

    if models.prod_model is not None:
        if not missing_esq:
            # v1.6.5: BYPASS conservador do classificador de desalinhamento no lado ESQ.
            # Motivo: em testes reais, a presença de mola ESQ ficou estável, mas o modelo
            # MobileNetV2 saturou p_ng_esq alto em praticamente todas as peças OK.
            # Mantemos a detecção de mola ausente no ESQ e o classificador de desalinhamento no DIR.
            prob_ng_esq = 0.0
            prob_ok_esq = 1.0
            cls_mis_esq = "OK"
            decision_band_esq = "LEFT_MISALIGN_BYPASS"
            margin_esq = -1.0
            attention_esq = False

        if not missing_dir:
            _, prob_ng_dir, probs_mis_dir = infer_mobilenetv2_prod(
                roi_dir_mis, models.prod_model, models.prod_class_names,
                models.prod_pos_idx, thr_ng_ng, img_size=models.prod_img_size
            )
            prob_ok_dir = float((probs_mis_dir or {}).get("OK", 1.0 - prob_ng_dir))
            cls_mis_dir, decision_band_dir, margin_dir, attention_dir = decide_misaligned_status(
                prob_ng_dir, prob_ok_dir, thr_ng_ok_dir, thr_ng_ng_dir, margin_abs=margin_abs
            )

    mis_esq = (cls_mis_esq == "NG_MISALIGNED") and (not missing_esq)
    mis_dir = (cls_mis_dir == "NG_MISALIGNED") and (not missing_dir)

    defect_esq = "NG_MISSING" if missing_esq else ("NG_MISALIGNED" if mis_esq else "OK")
    defect_dir = "NG_MISSING" if missing_dir else ("NG_MISALIGNED" if mis_dir else "OK")

    if defect_esq == "NG_MISSING" or defect_dir == "NG_MISSING":
        defect_type = "NG_MISSING"
    elif defect_esq == "NG_MISALIGNED" or defect_dir == "NG_MISALIGNED":
        defect_type = "NG_MISALIGNED"
    else:
        defect_type = "OK"

    return {
        "roi_esq": roi_esq,
        "roi_dir": roi_dir,
        "p_pres_esq": float(p_pres_esq),
        "p_pres_dir": float(p_pres_dir),
        "prob_ng_esq": float(prob_ng_esq),
        "prob_ng_dir": float(prob_ng_dir),
        "prob_ok_esq": float(prob_ok_esq),
        "prob_ok_dir": float(prob_ok_dir),
        "defect_esq": defect_esq,
        "defect_dir": defect_dir,
        "defect_type": defect_type,
        "ok_esq": defect_esq == "OK",
        "ok_dir": defect_dir == "OK",
        "aprovado": defect_type == "OK",
        "decision_band_esq": decision_band_esq,
        "decision_band_dir": decision_band_dir,
        "attention_flag": bool(attention_esq or attention_dir),
        "left_flip_for_misaligned": bool(left_flip_for_misaligned),
        "thr_presente": float(th_presente),
        "thr_ng_ok": float(thr_ng_ok),
        "thr_ng_ng": float(thr_ng_ng),
        "thr_ng_ok_esq": float(thr_ng_ok_esq),
        "thr_ng_ng_esq": float(thr_ng_ng_esq),
        "thr_ng_ok_dir": float(thr_ng_ok_dir),
        "thr_ng_ng_dir": float(thr_ng_ng_dir),
        "cls_pres_esq": cls_pres_esq,
        "cls_pres_dir": cls_pres_dir,
        "conf_pres_esq_argmax": float(conf_esq),
        "conf_pres_dir_argmax": float(conf_dir),
        "core_version": "v19-final-v8.1.2-stable-reset",
    }


def merge_temporal(results):
    if not results:
        raise RuntimeError("sem resultados temporais")
    last = dict(results[-1])

    p_pres_esq_vals = [float(r["p_pres_esq"]) for r in results]
    p_pres_dir_vals = [float(r["p_pres_dir"]) for r in results]
    prob_ng_esq_vals = [float(r["prob_ng_esq"]) for r in results]
    prob_ng_dir_vals = [float(r["prob_ng_dir"]) for r in results]
    prob_ok_esq_vals = [float(r.get("prob_ok_esq", 1.0 - float(r["prob_ng_esq"]))) for r in results]
    prob_ok_dir_vals = [float(r.get("prob_ok_dir", 1.0 - float(r["prob_ng_dir"]))) for r in results]

    last["p_pres_esq"] = float(np.mean(p_pres_esq_vals))
    last["p_pres_dir"] = float(np.mean(p_pres_dir_vals))
    last["prob_ng_esq"] = float(np.mean(prob_ng_esq_vals))
    last["prob_ng_dir"] = float(np.mean(prob_ng_dir_vals))
    last["prob_ok_esq"] = float(np.mean(prob_ok_esq_vals))
    last["prob_ok_dir"] = float(np.mean(prob_ok_dir_vals))

    th_pres = float(last["thr_presente"])
    # v1.6.5: na média temporal, respeitar thresholds específicos do lado DIR.
    # A v1.6.4 aplicava o threshold conservador no frame individual, mas a decisão temporal
    # recalculava usando o threshold global 0.60, provocando falsas reprovações no DIR.
    thr_ng_ok_dir = float(last.get("thr_ng_ok_dir", DEFAULT_THR_NG_OK_DIR))
    thr_ng_ng_dir = float(last.get("thr_ng_ng_dir", DEFAULT_THR_NG_NG_DIR))
    margin_abs = float(last.get("prod_margin_abs", 0.10))

    missing_esq = last["p_pres_esq"] < th_pres
    missing_dir = last["p_pres_dir"] < th_pres

    if missing_esq:
        defect_esq = "NG_MISSING"
        decision_band_esq = "MISSING"
        margin_esq = 0.0
        attention_esq = False
    else:
        # v1.6.5: mantém ESQ como OK quando a mola está presente;
        # desalinhamento ESQ fica temporariamente desabilitado até recalibração/re-treino.
        defect_esq = "OK"
        decision_band_esq = "LEFT_MISALIGN_BYPASS"
        margin_esq = -1.0
        attention_esq = False
        last["prob_ng_esq"] = 0.0
        last["prob_ok_esq"] = 1.0

    if missing_dir:
        defect_dir = "NG_MISSING"
        decision_band_dir = "MISSING"
        margin_dir = 0.0
        attention_dir = False
    else:
        defect_dir, decision_band_dir, margin_dir, attention_dir = decide_misaligned_status(
            last["prob_ng_dir"], last["prob_ok_dir"], thr_ng_ok_dir, thr_ng_ng_dir, margin_abs=margin_abs
        )

    if defect_esq == "NG_MISSING" or defect_dir == "NG_MISSING":
        defect_type = "NG_MISSING"
    elif defect_esq == "NG_MISALIGNED" or defect_dir == "NG_MISALIGNED":
        defect_type = "NG_MISALIGNED"
    else:
        defect_type = "OK"

    last.update({
        "defect_esq": defect_esq,
        "defect_dir": defect_dir,
        "defect_type": defect_type,
        "ok_esq": defect_esq == "OK",
        "ok_dir": defect_dir == "OK",
        "aprovado": defect_type == "OK",
        "decision_band_esq": decision_band_esq,
        "decision_band_dir": decision_band_dir,
        "margin_esq": float(margin_esq),
        "margin_dir": float(margin_dir),
        "attention_flag": bool(attention_esq or attention_dir),
        "temporal_smoothing_used": True,
        "temporal_n_frames": len(results),
        "temporal_p_pres_esq": p_pres_esq_vals,
        "temporal_p_pres_dir": p_pres_dir_vals,
        "temporal_p_ng_esq": prob_ng_esq_vals,
        "temporal_p_ng_dir": prob_ng_dir_vals,
        "temporal_p_ok_esq": prob_ok_esq_vals,
        "temporal_p_ok_dir": prob_ok_dir_vals,
    })
    return last


def read_fresh_frame(cap, flush_grabs=2, sleep_ms=5, extra_reads=1):
    for _ in range(int(flush_grabs)):
        cap.grab()
        time.sleep(max(0, sleep_ms) / 1000.0)
    frame = None
    for _ in range(max(1, int(extra_reads) + 1)):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    return frame


def inspect_once(cap, models, cfg, cycle_id):
    log(f"[{cycle_id:03d}] capture_begin")
    frame = read_fresh_frame(cap, flush_grabs=2, sleep_ms=5, extra_reads=1)
    if frame is None:
        raise RuntimeError("Falha ao capturar frame.")
    log(f"[{cycle_id:03d}] capture_ok shape={frame.shape}")

    # v8.1.2 stable reset: força 1 frame temporal para evitar inferência dupla e variação por config_molas.json
    n_frames = 1
    delay_ms = int(cfg.get("temporal_delay_ms", DEFAULT_TEMPORAL_DELAY_MS))

    results = [infer_dual_on_frame_v18(frame, models, cfg)]
    for _ in range(n_frames - 1):
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        f2 = read_fresh_frame(cap, flush_grabs=1, sleep_ms=delay_ms, extra_reads=0)
        if f2 is not None:
            results.append(infer_dual_on_frame_v18(f2, models, cfg))

    res = merge_temporal(results) if len(results) > 1 else results[0]
    final = res["defect_type"]

    log(
        f"[{cycle_id:03d}] RESULT={final} "
        f"ESQ={res['defect_esq']} DIR={res['defect_dir']} "
        f"p_pres_esq={res['p_pres_esq']:.3f} p_pres_dir={res['p_pres_dir']:.3f} "
        f"p_ng_esq={res['prob_ng_esq']:.3f} p_ng_dir={res['prob_ng_dir']:.3f} "
        f"p_ok_esq={res.get('prob_ok_esq', 0.0):.3f} p_ok_dir={res.get('prob_ok_dir', 0.0):.3f} "
        f"left_flip={res.get('left_flip_for_misaligned', False)} "
        f"band_esq={res.get('decision_band_esq','')} band_dir={res.get('decision_band_dir','')}"
    )

    ts_file = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = LOG_DIR / f"core_v16_{ts_file}_{final}.jpg"
    cv2.imwrite(str(out), frame)

    roi_esq_path = LOG_DIR / f"core_v16_{ts_file}_{final}_ROI_ESQ.jpg"
    roi_dir_path = LOG_DIR / f"core_v16_{ts_file}_{final}_ROI_DIR.jpg"
    try:
        if isinstance(res.get("roi_esq"), np.ndarray):
            cv2.imwrite(str(roi_esq_path), res["roi_esq"])
    except Exception:
        pass
    try:
        if isinstance(res.get("roi_dir"), np.ndarray):
            cv2.imwrite(str(roi_dir_path), res["roi_dir"])
    except Exception:
        pass

    status = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cycle": int(cycle_id),
        "result": final,
        "result_esq": res["defect_esq"],
        "result_dir": res["defect_dir"],
        "p_esq": float(res["p_pres_esq"]),
        "p_dir": float(res["p_pres_dir"]),
        "prob_ng_esq": float(res["prob_ng_esq"]),
        "prob_ng_dir": float(res["prob_ng_dir"]),
        "threshold": float(res["thr_presente"]),
        "thr_ng_ok": float(res["thr_ng_ok"]),
        "thr_ng_ng": float(res["thr_ng_ng"]),
        "thr_ng_ng_esq": float(res.get("thr_ng_ng_esq", res["thr_ng_ng"])),
        "thr_ng_ng_dir": float(res.get("thr_ng_ng_dir", res["thr_ng_ng"])),
        "image_path": str(out),
        "roi_esq_path": str(roi_esq_path) if roi_esq_path.exists() else "",
        "roi_dir_path": str(roi_dir_path) if roi_dir_path.exists() else "",
        "prob_ok_esq": float(res.get("prob_ok_esq", 1.0 - float(res["prob_ng_esq"]))),
        "prob_ok_dir": float(res.get("prob_ok_dir", 1.0 - float(res["prob_ng_dir"]))),
        "decision_band_esq": str(res.get("decision_band_esq", "")),
        "decision_band_dir": str(res.get("decision_band_dir", "")),
        "attention_flag": bool(res.get("attention_flag", False)),
        "aprovado": bool(res.get("aprovado", final == "OK")),
        "status": "OK",
        "core_version": "v19-final-v8.1.2-stable-reset",
        "note": "v1.6.2: presence mantido + guarda conservadora para NG_MISALIGNED esquerdo + thresholds por lado",
    }
    # v19 final: publicar campos completos para a interface Streamlit v18
    try:
        status.update({
            "defect_type": res.get("defect_type", final),
            "defect_esq": res.get("defect_esq", "OK"),
            "defect_dir": res.get("defect_dir", "OK"),
            "aprovado": bool(res.get("aprovado", final == "OK")),
            "ok_esq": bool(res.get("ok_esq", res.get("defect_esq") == "OK")),
            "ok_dir": bool(res.get("ok_dir", res.get("defect_dir") == "OK")),
            "prob_ok_esq": float(res.get("prob_ok_esq", 1.0 - float(res.get("prob_ng_esq", 0.0)))),
            "prob_ok_dir": float(res.get("prob_ok_dir", 1.0 - float(res.get("prob_ng_dir", 0.0)))),
            "decision_band_esq": res.get("decision_band_esq", ""),
            "decision_band_dir": res.get("decision_band_dir", ""),
            "attention_flag": bool(res.get("attention_flag", False)),
            "thr_presente": float(res.get("thr_presente", status.get("threshold", 0.5))),
            "thr_ng_ok_esq": float(res.get("thr_ng_ok_esq", status.get("thr_ng_ok", 0.45))),
            "thr_ng_ok_dir": float(res.get("thr_ng_ok_dir", status.get("thr_ng_ok", 0.45))),
            "thr_ng_ng_esq": float(res.get("thr_ng_ng_esq", status.get("thr_ng_ng", 0.60))),
            "thr_ng_ng_dir": float(res.get("thr_ng_ng_dir", status.get("thr_ng_ng", 0.60))),
            "left_flip_for_misaligned": bool(res.get("left_flip_for_misaligned", False)),
        })
    except Exception:
        pass

    summary = update_summary(final)
    status["summary"] = summary
    atomic_write_json(LAST_RESULT_TMP, LAST_RESULT_JSON, status)
    write_heartbeat(status="running", cycle=cycle_id)
    return final


def open_serial():
    if serial is None:
        raise RuntimeError("pyserial não disponível.")
    log(f"Abrindo serial {SERIAL_PORT} @ {SERIAL_BAUD}")
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.2)
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    return ser


def wait_for_present_one(ser):
    while True:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        val = parse_serial_line(line)
        if val is not None:
            log(f"serial={val} raw='{line}'")
            if val == 1:
                return


def wait_for_zero(ser):
    while True:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        val = parse_serial_line(line)
        if val is not None:
            log(f"serial={val} raw='{line}'")
            if val == 0:
                return


def drain_serial_input(ser, duration_s=0.10):
    """Descarta eventos seriais antigos após rearme/cancelamento."""
    end = time.time() + max(0.0, float(duration_s))
    try:
        while time.time() < end:
            raw = ser.readline()
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                line = str(raw)
            val = parse_serial_line(line)
            if val is not None:
                log(f"serial_drain={val} raw='{line}'")
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
    except Exception:
        pass


def settle_still_present_or_cancel(ser, settle_s):
    """Aguarda o settle monitorando retorno para 0.

    Se a mão/peça acionou o sensor rapidamente e saiu antes do fim do settle,
    cancela a captura para evitar foto da base vazia.
    Como o Arduino atual envia apenas mudanças de estado, se nenhum 0 chegar
    durante o settle assumimos que o cover continua presente.
    """
    deadline = time.time() + max(0.0, float(settle_s))
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        try:
            line = raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            line = str(raw)
        val = parse_serial_line(line)
        if val is not None:
            log(f"serial_settle={val} raw='{line}'")
            if SETTLE_CANCEL_ON_ZERO and val == 0:
                return False
    return True


def main():
    log("=== SVC CORE MOLAS V19 FINAL - V8.1.2 STABLE RESET ===")
    cfg = load_config()

    log("Abrindo câmera...")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir câmera.")

    models = IndustrialModels()
    ser = open_serial()
    cycle = 0

    log("Pronto. Aguardando sensor=1...")

    try:
        while (MAX_CYCLES <= 0) or (cycle < MAX_CYCLES):
            write_heartbeat(status="waiting_sensor", cycle=cycle)
            wait_for_present_one(ser)
            cycle += 1
            write_heartbeat(status="processing", cycle=cycle)

            log(f"[{cycle:03d}] sensor_triggered -> aguardando settle {SETTLE_S:.2f}s")
            if not settle_still_present_or_cancel(ser, SETTLE_S):
                log(f"[{cycle:03d}] CANCELADO: sensor voltou para 0 durante settle; não capturar base vazia.")
                # v8.1.1: NÃO drenar a serial após cancelamento.
                # Motivo: na inserção real pode ocorrer sequência 1->0->1
                # (mão/cover passando pelo sensor e depois cover assentado na base).
                # Se drenarmos aqui, podemos consumir justamente o PRESENT=1 correto
                # do cover já posicionado, e o sistema fica só incrementando ciclo sem capturar.
                log("Aguardando próxima borda PRESENT=1...")
                continue

            try:
                inspect_once(cap, models, cfg, cycle)
            except Exception as e:
                log(f"[{cycle:03d}] ERRO_INSPECAO: {e}")

            if REARM_ZERO_REQUIRED:
                log(f"[{cycle:03d}] aguardando sensor=0 para rearmar...")
                wait_for_zero(ser)
                drain_serial_input(ser, REARM_CLEAR_BUFFER_S)
                log(f"[{cycle:03d}] rearmado")

            log("Aguardando próxima peça...")

    finally:
        try:
            ser.close()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        write_heartbeat(status="stopped", cycle=cycle)
        log("Finalizado.")


if __name__ == "__main__":
    main()
