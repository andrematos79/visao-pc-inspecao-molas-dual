import sys
from typing import Optional

def beep_ng():
    # Windows
    if sys.platform.startswith("win"):
        try:
            import winsound
            winsound.Beep(1200, 180)   # freq Hz, dur ms
            winsound.Beep(900, 180)
        except Exception:
            pass
    else:
        # fallback (linux/mac) - pode não funcionar em todos
        print("\a", end="")
import threading

SENSOR_FIRE_EVENT = threading.Event()
import threading
MODEL_LOAD_LOCK = threading.Lock()
from pathlib import Path
import json
import re
import os
import csv
import io
import base64
import time
import random
import shutil
import platform
import subprocess
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import smtplib
import ssl
import mimetypes
from email.message import EmailMessage

import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
import re
import time

import re

def parse_serial_line(line: str):
    """
    Converte uma linha vinda do Arduino em present=True/False.

    Aceita:
      "1", "0"
      "P:1", "P:0"
      "SENSOR:1", "SENSOR:0"
      "present=1", "present=0"
      "PRESENT=1", "PRESENT=0"
    Retorna: True/False ou None se não reconhecer.
    """
    if not line:
        return None

    s = line.strip()

    # 1) normaliza (remove \r, \n e espaços)
    s_clean = s.replace("\r", "").replace("\n", "").strip()
    s_low = s_clean.lower()

    # 2) casos diretos "1"/"0"
    if s_low in ("1", "0"):
        return s_low == "1"

    # 3) padrões explícitos (robusto, não depende de estar no fim)
    #    Ex: "PRESENT=1", "present=0", "sensor:1", "p:0"
    m = re.search(r'\b(present|sensor|p)\s*[:=]\s*([01])\b', s_low)
    if m:
        return m.group(2) == "1"

    # 4) fallback: encontra QUALQUER 0/1 isolado na linha (último)
    bits = re.findall(r'\b[01]\b', s_low)
    if bits:
        return bits[-1] == "1"

    return None

# ==========================================================
# UI — Resultado Industrial (visual profissional)
# ==========================================================
def render_resultado_industrial(res: Optional[dict]) -> None:
    """Renderiza painel de resultado industrial com faixa de atenção para borda de decisão."""
    if not isinstance(res, dict):
        return

    def _map_def(def_code: str):
        if def_code == "OK":
            return "OK", "#22c55e"
        if def_code == "NG_MISSING":
            return "FALTANDO", "#dc2626"
        if def_code == "NG_MISALIGNED":
            return "DESALINHADA", "#f59e0b"
        return "NG", "#dc2626"

    final_code = str(res.get("defect_type") or ("OK" if res.get("aprovado", False) else "NG"))
    attention_flag = bool(res.get("attention_flag", False)) and final_code == "OK"

    if final_code == "OK" and attention_flag:
        titulo = "⚠ APROVADO COM ATENÇÃO"
        cor_box = "#fef3c7"
        cor_text = "#92400e"
        cor_border = "#f59e0b"
    else:
        is_ok = (final_code == "OK")
        titulo = "✔ APROVADO" if is_ok else "✖ REPROVADO"
        cor_box = "#dcfce7" if is_ok else "#fee2e2"
        cor_text = "#166534" if is_ok else "#7f1d1d"
        cor_border = "#22c55e" if is_ok else "#dc2626"

    esq_txt, esq_cor = _map_def(str(res.get("defect_esq", "OK")))
    dir_txt, dir_cor = _map_def(str(res.get("defect_dir", "OK")))

    p_pres_esq = res.get("p_pres_esq", None)
    p_pres_dir = res.get("p_pres_dir", None)
    p_ng_esq = res.get("prob_ng_esq", None)
    p_ng_dir = res.get("prob_ng_dir", None)
    thr_ng_ok = res.get("thr_ng_ok", None)
    thr_ng_ng = res.get("thr_ng_ng", None)
    band_esq = res.get("decision_band_esq", "-")
    band_dir = res.get("decision_band_dir", "-")

    def _fmt(v):
        try:
            return f"{float(v):.3f}"
        except Exception:
            return "-"

    details_esq = f"p(presente)={_fmt(p_pres_esq)}"
    details_dir = f"p(presente)={_fmt(p_pres_dir)}"
    if p_ng_esq is not None:
        details_esq += f" | p(NG)={_fmt(p_ng_esq)}"
    if p_ng_dir is not None:
        details_dir += f" | p(NG)={_fmt(p_ng_dir)}"
    if thr_ng_ok is not None and thr_ng_ng is not None:
        details_esq += f" | faixas={_fmt(thr_ng_ok)}/{_fmt(thr_ng_ng)}"
        details_dir += f" | faixas={_fmt(thr_ng_ok)}/{_fmt(thr_ng_ng)}"
    if band_esq:
        details_esq += f" | banda={band_esq}"
    if band_dir:
        details_dir += f" | banda={band_dir}"

    html = f"""
<div style="border-radius:12px;padding:16px;background:{cor_box};border:3px solid {cor_border};text-align:center;margin-top:-34px;">
  <div style="font-size:40px;font-weight:900;color:{cor_text};margin-bottom:10px;">{titulo}</div>
  <div style="display:flex;justify-content:center;gap:48px;font-size:20px;font-weight:800;">
    <div>ESQ<br><span style="display:inline-block;color:white;background:{esq_cor};padding:7px 18px;border-radius:8px;min-width:150px;">{esq_txt}</span></div>
    <div>DIR<br><span style="display:inline-block;color:white;background:{dir_cor};padding:7px 18px;border-radius:8px;min-width:150px;">{dir_txt}</span></div>
  </div>
  <div style="margin-top:10px;font-size:14px;font-weight:600;color:{cor_text};opacity:0.95;">
    ESQ: {details_esq}<br>
    DIR: {details_dir}
  </div>
</div>
"""

    st.markdown(html, unsafe_allow_html=True)

import textwrap
import threading
import queue
import serial
import serial.tools.list_ports
import os
import json
import numpy as np
import tensorflow as tf
# ==========================================================
# PRODUÇÃO — Carregar MobileNetV2 + pacote (classes/threshold)
# ==========================================================
# ==========================================================
# MODELO DE DESALINHAMENTO (PRODUÇÃO)
# Aqui carregamos o pacote treinado da MobileNetV2 usado
# para classificar cada ROI como:
#   - OK
#   - NG_MISALIGNED
#
# Ponto importante para ajustes futuros:
# - threshold bruto do pacote treinado
# - tamanho de entrada da imagem (img_size)
# - índice/classe positiva de NG
# ==========================================================
def load_production_package(outputs_dir: str):
    pkg_path = os.path.join(outputs_dir, "production_package.json")
    if not os.path.isfile(pkg_path):
        raise FileNotFoundError(f"production_package.json não encontrado em: {pkg_path}")

    with open(pkg_path, "r", encoding="utf-8") as f:
        pkg = json.load(f)

    class_names = pkg["class_names"]                     # ex: ["NG_MISALIGNED","OK"]
    pos_name = pkg["pos_class_name"]                     # "NG_MISALIGNED"
    pos_idx = int(pkg["pos_class_index"])                # 0
    thr = float(pkg["best_threshold_ng"]["thr"])         # 0.40

    img_size = tuple(pkg.get("img_size", [224, 224]))    # [224,224]
    return class_names, pos_name, pos_idx, thr, img_size


def load_mobilenetv2_prod_model(outputs_dir: str):
    model_path = os.path.join(outputs_dir, "model_final.keras")
    if not os.path.isfile(model_path):
        # fallback (se preferir usar best_model)
        alt = os.path.join(outputs_dir, "best_model.keras")
        if os.path.isfile(alt):
            model_path = alt
        else:
            raise FileNotFoundError(
                f"Modelo não encontrado. Esperado: {model_path} (ou best_model.keras)"
            )

    model = tf.keras.models.load_model(model_path)
    return model, model_path


# ==========================================================
# PRODUÇÃO — Inferência (retorna classe, prob_ng, probs)
# ==========================================================
# ==========================================================
# MODELO DE DESALINHAMENTO (PRODUÇÃO)
# Aqui carregamos o pacote treinado da MobileNetV2 usado
# para classificar cada ROI como:
#   - OK
#   - NG_MISALIGNED
#
# Ponto importante para ajustes futuros:
# - threshold bruto do pacote treinado
# - tamanho de entrada da imagem (img_size)
# - índice/classe positiva de NG
# ==========================================================
def infer_mobilenetv2_prod(
    bgr_img: np.ndarray,
    model: tf.keras.Model,
    class_names: list[str],
    pos_idx: int,
    thr_ng: float,
    img_size: tuple[int, int] = (224, 224),
):
    """
    bgr_img: frame BGR do OpenCV (H,W,3)
    Retorna:
      pred_label: "OK" ou "NG_MISALIGNED"
      prob_ng: probabilidade da classe NG
      probs: dict {"NG_MISALIGNED": p0, "OK": p1}
    """
    if bgr_img is None or bgr_img.size == 0:
        raise ValueError("Imagem vazia na inferência.")

    # BGR -> RGB
    rgb = bgr_img[..., ::-1]

    # Resize para input da MobileNetV2 (224x224)
    x = tf.image.resize(rgb, img_size, method="bilinear")
    x = tf.cast(x, tf.float32)
    x = tf.expand_dims(x, axis=0)  # (1,H,W,3)

    # O modelo já tem preprocess_input dentro do grafo (no seu script de treino),
    # então aqui NÃO fazemos preprocess_input de novo.
    p = model.predict(x, verbose=0)[0]  # softmax (2,)
    p = np.asarray(p, dtype=np.float32)

    prob_ng = float(p[pos_idx])
    probs = {class_names[i]: float(p[i]) for i in range(len(class_names))}

    pred_label = "NG_MISALIGNED" if prob_ng >= thr_ng else "OK"
    return pred_label, prob_ng, probs
# Donut (matplotlib) — com fallback se não existir
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


APP_VERSION = "v1.0.0"
APP_STAGE = "Stable"

# ==========================================================
# MES / RASTREABILIDADE
# ==========================================================
TRACE_LOG_PATH = None
MES_XML_DIR = None

def generate_inspection_id() -> str:
    return datetime.now().strftime("INSP_%Y%m%d_%H%M%S_%f")

def sanitize_filename(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r'[\/*?:"<>|\s]+', "_", s)
    return s[:120] if s else "SEM_DADO"

def validate_production_order(op: str, min_len: int = 4) -> tuple[bool, str]:
    op = (op or "").strip()
    if len(op) < min_len:
        return False, "Ordem de Produção inválida ou muito curta."
    return True, ""

def validate_equipment_id(equipment_id: str, min_len: int = 3) -> tuple[bool, str]:
    equipment_id = (equipment_id or "").strip()
    if len(equipment_id) < min_len:
        return False, "Equipment ID inválido ou muito curto."
    return True, ""

def validate_serial_qr(serial_code: str, min_len: int = 4) -> tuple[bool, str]:
    serial_code = (serial_code or "").strip()
    if len(serial_code) < min_len:
        return False, "Número de Série / QRCode inválido ou muito curto."
    return True, ""

def validate_operation_context() -> tuple[bool, str]:
    mes_enabled = bool(st.session_state.get("mes_enabled", False))
    traceability_enabled = bool(st.session_state.get("traceability_enabled", False))
    production_order = st.session_state.get("production_order", "")
    equipment_id = st.session_state.get("equipment_id", "")
    serial_qr_code = st.session_state.get("serial_qr_code", "")

    if mes_enabled:
        if not traceability_enabled:
            return False, "MES ativo exige rastreabilidade por Serial / QRCode."
        ok, msg = validate_production_order(production_order)
        if not ok:
            return False, msg
        ok, msg = validate_equipment_id(equipment_id)
        if not ok:
            return False, msg
        ok, msg = validate_serial_qr(serial_qr_code)
        if not ok:
            return False, msg
        return True, ""

    if traceability_enabled:
        ok, msg = validate_serial_qr(serial_qr_code)
        if not ok:
            return False, msg

    return True, ""

def create_inspection_xml(
    inspection_id: str,
    system_name: str,
    equipment_id: str,
    mes_enabled: bool,
    traceability_enabled: bool,
    production_order: str,
    serial_number: str,
    model_name: str,
    line_name: str,
    operation_mode: str,
    result_left: str,
    result_right: str,
    final_result: str,
    confidence_left: float,
    confidence_right: float,
    image_path: str,
    mes_status: str = "PENDENTE",
    source: str = "button",
) -> str:
    root = ET.Element("inspection")
    fields = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "inspection_id": inspection_id,
        "system_name": system_name,
        "equipment_id": equipment_id,
        "mes_enabled": str(bool(mes_enabled)).lower(),
        "traceability_enabled": str(bool(traceability_enabled)).lower(),
        "production_order": production_order or "",
        "serial_number": serial_number or "",
        "model_name": model_name or "",
        "line": str(line_name),
        "operation_mode": operation_mode or "",
        "source": source or "",
        "result_left": result_left or "",
        "result_right": result_right or "",
        "final_result": final_result or "",
        "confidence_left": f"{float(confidence_left):.6f}",
        "confidence_right": f"{float(confidence_right):.6f}",
        "image_path": image_path or "",
        "status_mes": mes_status or "PENDENTE",
    }
    for key, value in fields.items():
        child = ET.SubElement(root, key)
        child.text = str(value)

    xml_name = f"{inspection_id}_{sanitize_filename(serial_number) if serial_number else 'SEM_SERIAL'}.xml"
    xml_path = MES_XML_DIR / xml_name
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return str(xml_path)

def append_trace_log_csv(row: dict):
    fieldnames = [
        "timestamp", "inspection_id", "system_name", "equipment_id",
        "mes_enabled", "traceability_enabled", "production_order", "serial_number",
        "model_name", "line", "operation_mode", "source",
        "result_left", "result_right", "final_result",
        "confidence_left", "confidence_right",
        "image_path", "xml_path", "mes_status",
    ]
    file_exists = TRACE_LOG_PATH.exists()
    with open(TRACE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})



def normalize_serial_qr(serial: str) -> str:
    s = (serial or "").strip()
    s = s.replace("+", "-")
    s = re.sub(r"\s+", "", s)
    return s

def check_serial_duplicate(serial_number: str, csv_path: Path) -> bool:
    """Retorna True se o serial já existe no CSV de rastreabilidade."""
    serial_number = normalize_serial_qr(serial_number)
    if not serial_number or (csv_path is None) or (not Path(csv_path).exists()):
        return False

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                if normalize_serial_qr(row.get("serial_number", "")) == serial_number:
                    return True
    except Exception:
        return False

    return False

def render_production_dashboard() -> None:
    total = int(st.session_state.get("cnt_total", 0))
    ok = int(st.session_state.get("cnt_ok", 0))
    ng = int(st.session_state.get("cnt_ng", 0))
    yield_pct = (ok / total * 100.0) if total > 0 else 0.0

    op = str(st.session_state.get("production_order", "")).strip() or "---"
    model_name = str(st.session_state.get("selected_model_key", "MODELO_PADRAO"))
    line_name = str(st.session_state.get("line_name", "L01"))
    equipment_id = str(st.session_state.get("equipment_id", "SVC01")).strip() or "---"
    mes_txt = "ATIVO" if bool(st.session_state.get("mes_enabled", False)) else "DESLIGADO"
    trace_txt = "ATIVA" if bool(st.session_state.get("traceability_enabled", False)) else "DESLIGADA"

    st.markdown(f"""
    <div style="border:1px solid #d0d4d9;border-radius:10px;padding:5px 7px;margin:2px 0 8px 0;background:#f8fafc;">
      <div style="display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;justify-content:space-between;line-height:1.1;">
        <div style="font-size:11px;font-weight:700;color:#111827;">OP: <span style="font-weight:800;">{op}</span></div>
        <div style="font-size:11px;font-weight:700;color:#111827;">MODELO: <span style="font-weight:800;">{model_name}</span></div>
        <div style="font-size:11px;font-weight:700;color:#111827;">LINHA: <span style="font-weight:800;">{line_name}</span></div>
        <div style="font-size:11px;font-weight:700;color:#111827;">EQUIP.: <span style="font-weight:800;">{equipment_id}</span></div>
        <div style="font-size:11px;font-weight:700;color:#111827;">MES: <span style="font-weight:800;">{mes_txt}</span></div>
        <div style="font-size:11px;font-weight:700;color:#111827;">RASTREAB.: <span style="font-weight:800;">{trace_txt}</span></div>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;align-items:stretch;">
        <div style="min-width:72px;background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;padding:5px 7px;"><div style="font-size:10px;color:#6b7280;line-height:1;">TOTAL</div><div style="font-size:16px;font-weight:900;color:#111827;line-height:1.0;margin-top:3px;">{total}</div></div>
        <div style="min-width:72px;background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;padding:5px 7px;"><div style="font-size:10px;color:#6b7280;line-height:1;">OK</div><div style="font-size:16px;font-weight:900;color:#16a34a;line-height:1.0;margin-top:3px;">{ok}</div></div>
        <div style="min-width:72px;background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;padding:5px 7px;"><div style="font-size:10px;color:#6b7280;line-height:1;">NG</div><div style="font-size:16px;font-weight:900;color:#dc2626;line-height:1.0;margin-top:3px;">{ng}</div></div>
        <div style="min-width:92px;background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;padding:5px 7px;"><div style="font-size:10px;color:#6b7280;line-height:1;">YIELD</div><div style="font-size:16px;font-weight:900;color:#2563eb;line-height:1.0;margin-top:3px;">{yield_pct:.2f}%</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
# ==========================================================
# CONFIG / PATHS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
# MobileNetV2 PRODUÇÃO (OK vs NG_MISALIGNED)
# ==========================================================
PROD_MODEL_DIR = BASE_DIR / "models" / "mobilenetv2_prod"


LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
TRACE_LOG_PATH = LOG_DIR / "inspection_trace_log.csv"
MES_XML_DIR = LOG_DIR / "mes_xml"
MES_XML_DIR.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "modelo_molas.keras"
LABELS_PATH = BASE_DIR / "labels.json"
CONFIG_PATH = BASE_DIR / "config_molas.json"
REGISTRY_PATH = BASE_DIR / "models_registry.json"
EMAIL_CONFIG_PATH = BASE_DIR / "config_email.json"
EMAIL_CONTACTS_PATH = BASE_DIR / "email_contacts.json"
AUTO_REPORT_CONFIG_PATH = BASE_DIR / "config_relatorios_automaticos.json"
AUTO_REPORT_HISTORY_PATH = BASE_DIR / "historico_envio_relatorios.json"

IMG_SIZE = (224, 224)
DEFAULT_THRESH_PRESENTE = 0.80
DEFAULT_NORMALIZE_LAB = True
DEFAULT_THRESH_PRESENTE = 0.50
DEFAULT_THR_NG_OK = 0.45
DEFAULT_THR_NG_NG = 0.60

ENG_PIN = "1234"  # PIN DO MODO ENGENHARIA

DEFAULT_ROI = {
    "ESQ": {"x0": 8,  "x1": 35,  "y0": 10, "y1": 82},
    "DIR": {"x0": 74, "x1": 100, "y0": 17, "y1": 83},
}

# ==========================================================
# SESSION STATE — INIT (blindado)
# ==========================================================
def ss_init():
    ss = st.session_state

    # ---- Serial runtime
    if "serial_on" not in ss: ss.serial_on = False
    if "serial_port" not in ss: ss.serial_port = "COM4"
    if "serial_baud" not in ss: ss.serial_baud = 115200
    if "serial_thread" not in ss: ss.serial_thread = None
    if "serial_stop_evt" not in ss: ss.serial_stop_evt = None
    if "serial_q" not in ss: ss.serial_q = queue.Queue()
    if "serial_last_present" not in ss: ss.serial_last_present = None
    if "serial_last_trigger_ts" not in ss: ss.serial_last_trigger_ts = 0.0
    if "serial_status" not in ss: ss.serial_status = "OFF"

    if "serial_prev_present" not in ss: ss.serial_prev_present = None
    if "serial_lockout_until" not in ss: ss.serial_lockout_until = 0.0
    # ---- Serial trigger tuning
    if "serial_trigger_mode" not in ss: ss.serial_trigger_mode = "stable_high"  # default robusto
    if "serial_stable_ms" not in ss: ss.serial_stable_ms = 220
    if "serial_debounce_s" not in ss: ss.serial_debounce_s = 0.6
    if "serial_lockout_s" not in ss: ss.serial_lockout_s = 1.2
    if "serial_high_since" not in ss: ss.serial_high_since = 0.0
    if "serial_cycle_fired" not in ss: ss.serial_cycle_fired = False


    # ---- Trigger / actions
    if "pending_trigger" not in ss: ss.pending_trigger = False
    if "pending_trigger_src" not in ss: ss.pending_trigger_src = None
    if "last_sensor_fire_ts" not in ss: ss.last_sensor_fire_ts = 0.0
    if "last_sensor_fire_status" not in ss: ss.last_sensor_fire_status = ""
    if "last_sensor_fire_error" not in ss: ss.last_sensor_fire_error = ""

    # ---- Sensor job state machine (evita travas / rerun no meio da inferência)
    if "sensor_job_pending" not in ss: ss.sensor_job_pending = False
    if "sensor_job_kind" not in ss: ss.sensor_job_kind = None
    if "sensor_job_armed_ts" not in ss: ss.sensor_job_armed_ts = 0.0
    if "sensor_job_ready_at" not in ss: ss.sensor_job_ready_at = 0.0

    if "sensor_settle_ms" not in ss: ss.sensor_settle_ms = 220
    if "capture_busy" not in ss: ss.capture_busy = False
    if "capture_busy_since" not in ss: ss.capture_busy_since = 0.0

    # ---- KPI / Yield
    if "kpi_total" not in ss: ss.kpi_total = 0
    if "kpi_ok" not in ss: ss.kpi_ok = 0
    if "kpi_ng" not in ss: ss.kpi_ng = 0
    if "kpi_streak_ok" not in ss: ss.kpi_streak_ok = 0
    if "kpi_last_label" not in ss: ss.kpi_last_label = None

    # ---- Last results / errors
    if "last_result" not in ss: ss.last_result = None
    if "last_error" not in ss: ss.last_error = None
    if "last_warning" not in ss: ss.last_warning = None

ss_init()

def list_com_ports():
    ports = []
    for p in serial.tools.list_ports.comports():
        ports.append(p.device)
    return ports or ["COM4"]
def serial_reader_worker(port: str, baud: int, q: "queue.Queue", stop_evt: threading.Event):
    """Thread: lê Serial sem travar UI; reconecta com backoff e fecha corretamente."""
    ser = None
    last_emit = None

    while not stop_evt.is_set():
        try:
            # 1) abre (ou reabre) a porta
            if ser is None:
                try:
                    ser = serial.Serial(port, baudrate=baud, timeout=0.2)
                    try:
                        ser.reset_input_buffer()
                    except Exception:
                        pass
                    q.put(("status", f"Serial aberta em {port} @ {baud}", None))
                except Exception as e:
                    q.put(("error", f"open:{e}", None))
                    time.sleep(1.0)
                    continue

            # 2) lê uma linha (precisa do Arduino mandando \n)
            raw = ser.readline()
            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                line = str(raw)

            present = parse_serial_line(line)

            # 3) se reconheceu 0/1, emite evento só quando mudar (evita spam)
            if present is not None:
                if last_emit is None or present != last_emit:
                    last_emit = present
                    q.put(("present", 1 if bool(present) else 0, line))
            else:
                # opcional: debug do que está chegando
                q.put(("raw", line, time.time()))

        except Exception as e:
            q.put(("error", f"read:{e}", None))
            try:
                if ser is not None:
                    ser.close()
            except Exception:
                pass
            ser = None
            time.sleep(0.5)

    # shutdown
    try:
        if ser is not None and ser.is_open:
            ser.close()
    except Exception:
        pass

def serial_start():
    ss = st.session_state
    if ss.serial_thread and ss.serial_thread.is_alive():
        # já está rodando
        ss.serial_on = True
        ss.serial_status = "ON"
        return

    # ✅ Reset do estado do gatilho ao ligar o Serial (evita lockout travado / estado preso)
    ss.serial_last_present = None
    ss.serial_prev_present = None
    ss.sensor_present = False
    ss.serial_high_since = 0.0
    ss.serial_cycle_fired = False
    ss.serial_last_trigger_ts = 0.0
    ss.serial_lockout_until = 0.0
    ss.pending_trigger = False
    ss.pending_trigger_src = None
    ss.sensor_job_pending = False
    ss.sensor_job_kind = None
    ss.sensor_job_armed_ts = 0.0

    # limpa fila (eventos antigos)
    try:
        q = ss.get("serial_q", None)
        if q is not None:
            while True:
                q.get_nowait()
    except Exception:
        pass

    ss.serial_stop_evt = threading.Event()
    ss.serial_thread = threading.Thread(
        target=serial_reader_worker,
        args=(ss.serial_port, ss.serial_baud, ss.serial_q, ss.serial_stop_evt),
        daemon=True
    )
    ss.serial_thread.start()
    try:
        ss.serial_q.put(("status", "thread_started", None))
    except Exception:
        pass
    ss.serial_status = "ON"
    ss.serial_on = True

def serial_stop():
    ss = st.session_state
    if ss.serial_stop_evt:
        ss.serial_stop_evt.set()
    ss.serial_status = "OFF"
    ss.serial_on = False


def poll_serial_events_and_maybe_trigger():
    """Consome eventos do Serial (via fila) e arma pending_trigger.

    Modos de disparo (ss.serial_trigger_mode):
      - 'release_1to0'  : borda 1->0 (soltar)
      - 'press_0to1'    : borda 0->1 (apertar/chegar)
      - 'stable_high'   : dispara 1x quando PRESENT=1 fica estável por N ms, e só rearma quando voltar a 0

    Importante:
      - Alguns firmwares só enviam linha quando o estado muda (sem repetição).
        Nesse caso, o modo 'stable_high' precisa de uma verificação por tempo,
        mesmo sem novos eventos na fila. Esta função faz isso.
      - Este poll NÃO dá st.rerun(); quem atualiza o loop é o auto-refresh.
    """
    ss = st.session_state
    q = ss.get("serial_q", None)
    if q is None:
        return

    mode = ss.get("serial_trigger_mode", "stable_high")
    stable_ms = int(ss.get("serial_stable_ms", 220))
    debounce_s = float(ss.get("serial_debounce_s", 0.6))
    lockout_s = float(ss.get("serial_lockout_s", 1.2))

    now = time.time()

    # Helper: checa lockout/debounce e arma trigger
    def _arm_trigger():
        nonlocal now
        if now < float(ss.get("serial_lockout_until", 0.0)):
            return False
        if (now - float(ss.get("serial_last_trigger_ts", 0.0))) < debounce_s:
            return False
        ss.serial_last_trigger_ts = now
        ss.serial_lockout_until = now + lockout_s
        ss.pending_trigger = True
        ss.pending_trigger_src = "sensor"
        return True

    # 1) Consome tudo que chegou na fila (mudanças de estado / erros)
    while True:
        try:
            evt = q.get_nowait()
        except Exception:
            break  # fila vazia

        if not evt:
            continue

        kind = evt[0]

        if kind in ("present", "sensor"):
            present = int(evt[1]) if kind=="present" else (1 if bool(evt[1]) else 0)
            ss.serial_last_present = present

            # debug line
            try:
                ss.serial_last_line = str(evt[2])
            except Exception:
                pass

            ss.sensor_present = (present == 1)

            prev = ss.get("serial_prev_present", None)
            ss.serial_prev_present = present

            now = time.time()

            # Rearme por ciclo: quando PRESENT volta a 0, libera novo disparo em qualquer modo
            if present == 0:
                ss.serial_cycle_fired = False
                ss.serial_high_since = 0.0

            # MODE: release_1to0 (dispara ao SOLTAR: 1 -> 0)
            if mode == "release_1to0":
                if prev == 1 and present == 0:
                    _arm_trigger()

            # MODE: press_0to1 (dispara ao CHEGAR: qualquer entrada em 1, 1x por ciclo)
            elif mode == "press_0to1":
                # Robusto: alguns cenários não fornecem a borda (prev pode ficar "preso" em 1).
                # Então garantimos 1 disparo por ciclo enquanto PRESENT=1.
                if present == 1 and not bool(ss.get("serial_cycle_fired", False)):
                    if _arm_trigger():
                        ss.serial_cycle_fired = True


                # ✅ Rearme obrigatório: só permite novo disparo quando PRESENT voltar a 0
                if present == 0:
                    ss.serial_cycle_fired = False
                    ss.serial_high_since = 0.0

            # MODE: stable_high (dispara 1x quando fica em 1 por N ms; rearma ao voltar a 0)
            else:
                if prev != 1 and present == 1:
                    ss.serial_high_since = now
                    ss.serial_cycle_fired = False

                if present == 0:
                    ss.serial_cycle_fired = False
                    ss.serial_high_since = 0.0
        elif kind == "error":
            ss.serial_status = f"ERR: {evt[1]}"
            ss.serial_on = False
            try:
                if ss.get("serial_stop_evt") is not None:
                    ss.serial_stop_evt.set()
            except Exception:
                pass

        else:
            try:
                ss.serial_last_line = str(evt[2] if len(evt) > 2 else evt)
            except Exception:
                pass

    # 2) Verificação por tempo (necessária quando o firmware NÃO repete PRESENT=1 continuamente)
    if mode == "stable_high":
        # se está presente e ainda não disparou nesse ciclo, verifica tempo
        if bool(ss.get("sensor_present", False)) and not bool(ss.get("serial_cycle_fired", False)):
            high_since = float(ss.get("serial_high_since", now))
            now = time.time()
            if (now - high_since) >= (stable_ms / 1000.0):
                if _arm_trigger():
                    ss.serial_cycle_fired = True


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
    st.session_state.setdefault("last_warning", None)
    st.session_state.setdefault("last_result", None)

    # frames
    st.session_state.setdefault("display_frame", None)
    st.session_state.setdefault("last_frame", None)
    st.session_state.setdefault("frozen", False)
    st.session_state.setdefault("frozen_frame", None)

    # assinatura p/ detecção de troca de peça (auto)
    st.session_state.setdefault("last_infer_sig", None)
    st.session_state.setdefault("last_infer_ts", 0.0)
    st.session_state.setdefault("live_sig", None)
    # ajustes do trigger por imagem
    st.session_state.setdefault("serial_min_interval_s", 0.8)
    st.session_state.setdefault("serial_image_diff_thr", 6.0)


    # câmera
    st.session_state.setdefault("cap", None)
    st.session_state.setdefault("camera_on", False)
    st.session_state.setdefault("cam_index_last", 0)

    # modelo/labels
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("labels", None)

    # MobileNetV2 Produção (OK vs NG_MISALIGNED)
    st.session_state.setdefault("use_mnv2_prod", True)
    st.session_state.setdefault("prod_model", None)
    st.session_state.setdefault("prod_model_path", None)
    st.session_state.setdefault("prod_class_names", None)
    st.session_state.setdefault("prod_pos_idx", None)
    st.session_state.setdefault("prod_thr_ng", None)
    st.session_state.setdefault("prod_thr_ng_ok", DEFAULT_THR_NG_OK)
    st.session_state.setdefault("prod_thr_ng_ng", DEFAULT_THR_NG_NG)
    st.session_state.setdefault("prod_margin_abs", 0.10)
    st.session_state.setdefault("prod_img_size", (224, 224))

    # modo + PIN
    st.session_state.setdefault("user_mode", "OPERADOR")
    st.session_state.setdefault("eng_unlocked", False)

    # modelo selecionado
    st.session_state.setdefault("selected_model_key", "MODELO_PADRAO")
    st.session_state.setdefault("product_model", st.session_state.get("selected_model_key", "MODELO_PADRAO"))

    # linha de produção
    st.session_state.setdefault("line_name", "L01")

    # MES / rastreabilidade
    st.session_state.setdefault("mes_enabled", False)
    st.session_state.setdefault("traceability_enabled", False)
    st.session_state.setdefault("production_order", "")
    st.session_state.setdefault("serial_qr_code", "")
    st.session_state.setdefault("equipment_id", "SVC01")
    st.session_state.setdefault("system_name", "SVC Inspeção de Molas - DUAL")
    st.session_state.setdefault("last_xml_path", "")
    st.session_state.setdefault("last_mes_status", "LOCAL")
    st.session_state.setdefault("last_inspection_id", "")

    # aprendizado
    st.session_state.setdefault("learning_last_saved", None)

    # relatórios
    st.session_state.setdefault("production_started_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.session_state.setdefault("last_report_pdf", "")
    st.session_state.setdefault("last_report_html", "")
    st.session_state.setdefault("last_report_generated_at", "")

    # e-mail / SMTP
    st.session_state.setdefault("email_reports_enabled", False)
    st.session_state.setdefault("email_send_on_generate", False)
    st.session_state.setdefault("email_auto_daily_enabled", False)
    st.session_state.setdefault("email_daily_time", "17:30")
    st.session_state.setdefault("email_to", "")
    st.session_state.setdefault("email_cc", "")
    st.session_state.setdefault("email_bcc", "")
    st.session_state.setdefault("email_subject_prefix", "[SVC] Relatório de Auditoria")
    st.session_state.setdefault("email_sender_name", "SVC Inspeção de Molas")
    st.session_state.setdefault("smtp_server", "smtp.office365.com")
    st.session_state.setdefault("smtp_port", 587)
    st.session_state.setdefault("smtp_user", "")
    st.session_state.setdefault("smtp_password", "")
    st.session_state.setdefault("smtp_use_tls", True)
    st.session_state.setdefault("email_config_loaded", False)
    st.session_state.setdefault("email_config_reload_pending", False)
    st.session_state.setdefault("email_config_reload_notice", "")
    st.session_state.setdefault("email_bootstrap_ok", False)
    st.session_state.setdefault("email_last_saved_at", "")
    st.session_state.setdefault("email_last_save_error", "")

    # automação de relatórios por JSON
    st.session_state.setdefault("auto_reports_enabled", False)
    st.session_state.setdefault("auto_reports_cfg_loaded", False)
    st.session_state.setdefault("auto_reports_status_msg", "")
    st.session_state.setdefault("auto_reports_last_check", "")

    # simulação por upload (modo Engenharia)
    st.session_state.setdefault("upload_test_frame", None)
    st.session_state.setdefault("upload_test_name", "")
    st.session_state.setdefault("upload_test_count_kpi", False)

    # evidências / auditoria
    st.session_state.setdefault("evidence_auto_enabled", True)
    st.session_state.setdefault("evidence_save_ok_limit", True)
    st.session_state.setdefault("evidence_retention_enabled", True)
    st.session_state.setdefault("evidence_retention_days", 60)
    st.session_state.setdefault("evidence_warning_gb", 5.0)
    st.session_state.setdefault("evidence_last_cleanup_ts", 0.0)
    st.session_state.setdefault("evidence_last_cleanup_files", 0)
    st.session_state.setdefault("evidence_last_cleanup_bytes", 0)
    st.session_state.setdefault("evidence_last_saved", None)
    st.session_state.setdefault("evidence_last_manual", None)
    st.session_state.setdefault("evidence_last_auto_signature", None)
    st.session_state.setdefault("evidence_last_auto_ts", 0.0)

    # auditoria detalhada
    st.session_state.setdefault("cnt_missing_esq", 0)
    st.session_state.setdefault("cnt_missing_dir", 0)
    st.session_state.setdefault("cnt_missing_both", 0)
    st.session_state.setdefault("cnt_misaligned_esq", 0)
    st.session_state.setdefault("cnt_misaligned_dir", 0)
    st.session_state.setdefault("cnt_misaligned_both", 0)
    st.session_state.setdefault("cnt_misto", 0)
    st.session_state.setdefault("cnt_ok_attention", 0)

    # coleta manual detalhada para dataset (modo Engenharia)
    st.session_state.setdefault("manual_cnt_ok_perfeita", 0)
    st.session_state.setdefault("manual_cnt_desalinhada_esq", 0)
    st.session_state.setdefault("manual_cnt_desalinhada_dir", 0)
    st.session_state.setdefault("manual_cnt_desalinhada_both", 0)
    st.session_state.setdefault("manual_cnt_faltando_esq", 0)
    st.session_state.setdefault("manual_cnt_faltando_dir", 0)
    st.session_state.setdefault("manual_cnt_faltando_both", 0)
    st.session_state.setdefault("manual_cnt_misto_des_esq_falt_dir", 0)
    st.session_state.setdefault("manual_cnt_misto_falt_esq_des_dir", 0)
    st.session_state.setdefault("manual_last_saved_detail", "")

init_session()

# ---- Sensor flags (Arduino) — usados pelo gatilho no Streamlit
st.session_state.setdefault("sensor_trigger", False)
st.session_state.setdefault("sensor_present", False)
st.session_state.setdefault("serial_last_line", "")


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


def load_email_config() -> dict:
    try:
        if EMAIL_CONFIG_PATH.exists():
            data = json.loads(EMAIL_CONFIG_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def save_email_config(payload: dict) -> None:
    EMAIL_CONFIG_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_email_contacts() -> dict:
    try:
        if EMAIL_CONTACTS_PATH.exists():
            data = json.loads(EMAIL_CONTACTS_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def save_email_contacts(payload: dict) -> None:
    EMAIL_CONTACTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _join_email_values(values) -> str:
    if isinstance(values, list):
        return "; ".join([str(x).strip() for x in values if str(x).strip()])
    return str(values or "").strip()


def apply_email_contacts_to_session(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        return
    if "to" in cfg:
        st.session_state["email_to"] = _join_email_values(cfg.get("to", []))
    if "cc" in cfg:
        st.session_state["email_cc"] = _join_email_values(cfg.get("cc", []))
    if "bcc" in cfg:
        st.session_state["email_bcc"] = _join_email_values(cfg.get("bcc", []))


def collect_email_contacts_from_session() -> dict:
    return {
        "to": [x.strip() for x in re.split(r'[;,]+', str(st.session_state.get("email_to", ""))) if x.strip()],
        "cc": [x.strip() for x in re.split(r'[;,]+', str(st.session_state.get("email_cc", ""))) if x.strip()],
        "bcc": [x.strip() for x in re.split(r'[;,]+', str(st.session_state.get("email_bcc", ""))) if x.strip()],
    }


def apply_email_config_to_session(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        return
    st.session_state["email_reports_enabled"] = bool(cfg.get("email_reports_enabled", False))
    st.session_state["email_send_on_generate"] = bool(cfg.get("email_send_on_generate", False))
    st.session_state["email_auto_daily_enabled"] = bool(cfg.get("email_auto_daily_enabled", False))
    st.session_state["email_daily_time"] = str(cfg.get("email_daily_time", "17:30"))
    st.session_state["email_subject_prefix"] = str(cfg.get("email_subject_prefix", "[SVC] Relatório de Auditoria"))
    st.session_state["email_sender_name"] = str(cfg.get("email_sender_name", "SVC Inspeção de Molas"))
    st.session_state["smtp_server"] = str(cfg.get("smtp_server", "smtp.office365.com"))
    try:
        st.session_state["smtp_port"] = int(cfg.get("smtp_port", 587))
    except Exception:
        st.session_state["smtp_port"] = 587
    st.session_state["smtp_user"] = str(cfg.get("smtp_user", ""))
    st.session_state["smtp_password"] = str(cfg.get("smtp_password", ""))
    st.session_state["smtp_use_tls"] = bool(cfg.get("smtp_use_tls", True))


def collect_email_config_from_session() -> dict:
    return {
        "email_reports_enabled": bool(st.session_state.get("email_reports_enabled", False)),
        "email_send_on_generate": bool(st.session_state.get("email_send_on_generate", False)),
        "email_auto_daily_enabled": bool(st.session_state.get("email_auto_daily_enabled", False)),
        "email_daily_time": str(st.session_state.get("email_daily_time", "17:30")),
        "email_subject_prefix": str(st.session_state.get("email_subject_prefix", "[SVC] Relatório de Auditoria")).strip(),
        "email_sender_name": str(st.session_state.get("email_sender_name", "SVC Inspeção de Molas")).strip(),
        "smtp_server": str(st.session_state.get("smtp_server", "smtp.office365.com")).strip(),
        "smtp_port": int(st.session_state.get("smtp_port", 587)),
        "smtp_user": str(st.session_state.get("smtp_user", "")).strip(),
        "smtp_password": str(st.session_state.get("smtp_password", "")).strip(),
        "smtp_use_tls": bool(st.session_state.get("smtp_use_tls", True)),
    }


def bootstrap_email_settings() -> None:
    cfg = load_email_config()
    contacts = load_email_contacts()

    if not contacts:
        migrated = {
            "to": [x.strip() for x in re.split(r'[;,]+', str(cfg.get("email_to", ""))) if x.strip()],
            "cc": [x.strip() for x in re.split(r'[;,]+', str(cfg.get("email_cc", ""))) if x.strip()],
            "bcc": [x.strip() for x in re.split(r'[;,]+', str(cfg.get("email_bcc", ""))) if x.strip()],
        }
        if migrated["to"] or migrated["cc"] or migrated["bcc"]:
            contacts = migrated
            try:
                save_email_contacts(contacts)
            except Exception:
                pass

    apply_email_config_to_session(cfg)
    apply_email_contacts_to_session(contacts)
    st.session_state["email_config_loaded"] = True
    st.session_state["email_bootstrap_ok"] = bool(
        st.session_state.get("smtp_user") or st.session_state.get("email_to") or st.session_state.get("email_cc") or st.session_state.get("email_bcc")
    )


def email_status_summary() -> str:
    enabled = bool(st.session_state.get("email_reports_enabled", False))
    if not enabled:
        return "E-mail: DESLIGADO"
    to_count = len([x for x in re.split(r'[;,]+', str(st.session_state.get("email_to", ""))) if x.strip()])
    return f"E-mail: ATIVO | Destinatários: {to_count}"


def persist_email_settings_if_needed(force: bool = False) -> bool:
    """Salva automaticamente as configurações/contatos de e-mail em JSON.
    Retorna True quando houve gravação em disco.
    """
    try:
        cfg = collect_email_config_from_session()
        contacts = collect_email_contacts_from_session()

        current_cfg = load_email_config()
        current_contacts = load_email_contacts()

        if force or cfg != current_cfg:
            save_email_config(cfg)
            changed_cfg = True
        else:
            changed_cfg = False

        if force or contacts != current_contacts:
            save_email_contacts(contacts)
            changed_contacts = True
        else:
            changed_contacts = False

        if changed_cfg or changed_contacts:
            st.session_state["email_bootstrap_ok"] = True
            st.session_state["email_last_saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return True
    except Exception as e:
        st.session_state["email_last_save_error"] = str(e)
    return False

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

    st.session_state["threshold_presente"] = float(cfg.get("threshold_presente", DEFAULT_THRESH_PRESENTE))
    st.session_state["threshold_ng_ok"] = float(cfg.get("threshold_ng_ok", DEFAULT_THR_NG_OK))
    st.session_state["threshold_ng_ng"] = float(cfg.get("threshold_ng_ng", DEFAULT_THR_NG_NG))
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

# ==========================================================
# CONFIGURAÇÃO PERSISTENTE DO MODELO
# Este bloco salva no JSON os parâmetros ajustados na engenharia.
#
# Parâmetros críticos para calibração em campo:
# - threshold_presente : define falta de mola
# - threshold_ng_ok    : acima disso a peça sai da zona de OK seguro
# - threshold_ng_ng    : acima disso classifica como desalinhada
# - threshold_margem   : margem mínima entre classes para decisão robusta
#
# Se o comportamento mudar na fábrica, começar a investigação por aqui.
# ==========================================================
def collect_config_from_session() -> dict:
    """
    Gera payload para salvar no JSON do modelo atual.
    """
    return {
        "threshold_presente": float(st.session_state.get("threshold_presente", DEFAULT_THRESH_PRESENTE)),
        "threshold_ng_ok": float(st.session_state.get("threshold_ng_ok", DEFAULT_THR_NG_OK)),
        "threshold_ng_ng": float(st.session_state.get("threshold_ng_ng", DEFAULT_THR_NG_NG)),
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
if ("threshold_presente" not in st.session_state) or ("threshold_ng_ok" not in st.session_state) or ("threshold_ng_ng" not in st.session_state):
    apply_config_to_session(get_effective_config(st.session_state.get("selected_model_key", "MODELO_PADRAO")))

if not st.session_state.get("email_config_loaded", False):
    bootstrap_email_settings()

if st.session_state.get("email_config_reload_pending", False):
    bootstrap_email_settings()
    st.session_state["email_config_reload_pending"] = False
    st.session_state["email_config_reload_notice"] = "Configuração de e-mail e contatos recarregada ✅"

# ==========================================================
# DATASET (APRENDIZADO) — CAPTURA E SALVAMENTO
# ==========================================================
DATASET_ROOT = BASE_DIR / "dataset_products"
DATASET_ROOT.mkdir(exist_ok=True)

# ==========================================================
# EVIDÊNCIAS / AUDITORIA / RETENÇÃO
# ==========================================================
EVIDENCE_DIR = BASE_DIR / "dataset_coleta_industrial"
AUTO_EVIDENCE_DIR = BASE_DIR / "dataset_auto_evidencias"
AUDIT_LOG_PATH = LOG_DIR / "evidence_audit_log.csv"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MANUAL_EVIDENCE_CLASSES = {
    "OK": EVIDENCE_DIR / "OK",
    "NG_DESALINHADO": EVIDENCE_DIR / "NG_DESALINHADO",
    "NG_FALTANDO": EVIDENCE_DIR / "NG_FALTANDO",
    "NG_MISTO": EVIDENCE_DIR / "NG_MISTO",
}

AUTO_EVIDENCE_CLASSES = {
    "OK_LIMITE": AUTO_EVIDENCE_DIR / "OK_LIMITE",
    "NG_DESALINHADO": AUTO_EVIDENCE_DIR / "NG_DESALINHADO",
    "NG_FALTANDO": AUTO_EVIDENCE_DIR / "NG_FALTANDO",
}

for _p in list(MANUAL_EVIDENCE_CLASSES.values()) + list(AUTO_EVIDENCE_CLASSES.values()):
    _p.mkdir(parents=True, exist_ok=True)

def map_defect_label(defect_type: str) -> str:
    defect_type = str(defect_type or "").strip().upper()
    if defect_type == "NG_MISSING":
        return "NG_FALTANDO"
    if defect_type == "NG_MISALIGNED":
        return "NG_DESALINHADO"
    return "OK"

def bytes_to_human(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(0, int(n_bytes or 0)))
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0

def folder_size_bytes(folder: Path) -> int:
    total = 0
    if folder.exists():
        for p in folder.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
    return total

def count_evidence_files(folder: Path, suffixes=(".jpg", ".jpeg", ".png", ".bmp", ".json")) -> int:
    total = 0
    if folder.exists():
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in suffixes:
                total += 1
    return total

def list_recent_files(folder: Path, limit: int = 8) -> list[Path]:
    files = []
    if folder.exists():
        for p in folder.rglob("*"):
            if p.is_file():
                try:
                    files.append((p.stat().st_mtime, p))
                except Exception:
                    pass
    files.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in files[:limit]]


def get_disk_status(path: Path) -> dict:
    try:
        usage = shutil.disk_usage(str(path))
        total_gb = usage.total / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        return {
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "total_gb": float(total_gb),
            "used_gb": float(used_gb),
            "free_gb": float(free_gb),
        }
    except Exception:
        return {
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "total_gb": 0.0,
            "used_gb": 0.0,
            "free_gb": 0.0,
        }


def disk_free_status_label(free_gb: float, warn_gb: float = 10.0, critical_gb: float = 5.0) -> tuple[str, str]:
    if free_gb <= critical_gb:
        return "critical", "CRÍTICO"
    if free_gb <= warn_gb:
        return "warning", "ATENÇÃO"
    return "ok", "NORMAL"


def defect_to_pt(code: str) -> str:
    code = str(code or "OK").strip().upper()
    mapping = {
        "OK": "OK",
        "NG_MISSING": "FALTANDO",
        "NG_MISALIGNED": "DESALINHADA",
    }
    return mapping.get(code, code)


def build_defect_detail_code(res: dict | None) -> str:
    res = res or {}
    de = str(res.get("defect_esq", "OK") or "OK").strip().upper()
    dd = str(res.get("defect_dir", "OK") or "OK").strip().upper()

    if de == "OK" and dd == "OK":
        return "OK_ATENCAO" if bool(res.get("attention_flag", False)) else "OK"
    if de == "NG_MISSING" and dd == "OK":
        return "FALTANDO_ESQ"
    if de == "OK" and dd == "NG_MISSING":
        return "FALTANDO_DIR"
    if de == "NG_MISSING" and dd == "NG_MISSING":
        return "FALTANDO_BOTH"
    if de == "NG_MISALIGNED" and dd == "OK":
        return "DESALINHADA_ESQ"
    if de == "OK" and dd == "NG_MISALIGNED":
        return "DESALINHADA_DIR"
    if de == "NG_MISALIGNED" and dd == "NG_MISALIGNED":
        return "DESALINHADA_BOTH"

    left = defect_to_pt(de)
    right = defect_to_pt(dd)
    return f"MISTO_{left}_ESQ_{right}_DIR"


def get_audit_counts_from_session() -> dict:
    return {
        "faltando_esq": int(st.session_state.get("cnt_missing_esq", 0)),
        "faltando_dir": int(st.session_state.get("cnt_missing_dir", 0)),
        "faltando_both": int(st.session_state.get("cnt_missing_both", 0)),
        "desalinhada_esq": int(st.session_state.get("cnt_misaligned_esq", 0)),
        "desalinhada_dir": int(st.session_state.get("cnt_misaligned_dir", 0)),
        "desalinhada_both": int(st.session_state.get("cnt_misaligned_both", 0)),
        "misto": int(st.session_state.get("cnt_misto", 0)),
        "ok_atencao": int(st.session_state.get("cnt_ok_attention", 0)),
    }


MANUAL_DETAIL_COUNTER_KEYS = {
    "OK": "manual_cnt_ok_perfeita",
    "DESALINHADA_ESQ": "manual_cnt_desalinhada_esq",
    "DESALINHADA_DIR": "manual_cnt_desalinhada_dir",
    "DESALINHADA_BOTH": "manual_cnt_desalinhada_both",
    "FALTANDO_ESQ": "manual_cnt_faltando_esq",
    "FALTANDO_DIR": "manual_cnt_faltando_dir",
    "FALTANDO_BOTH": "manual_cnt_faltando_both",
    "MISTO_DESALINHADA_ESQ_FALTANDO_DIR": "manual_cnt_misto_des_esq_falt_dir",
    "MISTO_FALTANDO_ESQ_DESALINHADA_DIR": "manual_cnt_misto_falt_esq_des_dir",
}


def inc_manual_detail_counter(detail_code: str) -> None:
    key = MANUAL_DETAIL_COUNTER_KEYS.get(str(detail_code or '').strip().upper())
    if key:
        st.session_state[key] = int(st.session_state.get(key, 0)) + 1


def get_manual_detail_counts() -> dict:
    return {
        detail: int(st.session_state.get(key, 0))
        for detail, key in MANUAL_DETAIL_COUNTER_KEYS.items()
    }


def manual_detail_human(detail_code: str) -> str:
    mapping = {
        "OK": "OK (ambas perfeitas)",
        "DESALINHADA_ESQ": "Desalinhado ESQ / DIR OK",
        "DESALINHADA_DIR": "ESQ OK / DIR desalinhado",
        "DESALINHADA_BOTH": "Ambos desalinhados",
        "FALTANDO_ESQ": "ESQ faltando / DIR OK",
        "FALTANDO_DIR": "ESQ OK / DIR faltando",
        "FALTANDO_BOTH": "Ambas faltando",
        "MISTO_DESALINHADA_ESQ_FALTANDO_DIR": "ESQ desalinhado / DIR faltando",
        "MISTO_FALTANDO_ESQ_DESALINHADA_DIR": "ESQ faltando / DIR desalinhado",
    }
    return mapping.get(str(detail_code or '').strip().upper(), str(detail_code or ''))


def manual_label_from_detail(detail_code: str) -> str:
    dc = str(detail_code or '').strip().upper()
    if dc == 'OK':
        return 'OK'
    if dc.startswith('DESALINHADA_'):
        return 'NG_DESALINHADO'
    if dc.startswith('FALTANDO_'):
        return 'NG_FALTANDO'
    if dc.startswith('MISTO_'):
        return 'NG_MISTO'
    return 'OK'

def draw_roi_overlay(frame_bgr: np.ndarray) -> np.ndarray:
    img = frame_bgr.copy()
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]

    def _box(x0p, x1p, y0p, y1p, color, label):
        x0 = int(max(0.0, min(1.0, x0p / 100.0)) * w)
        x1 = int(max(0.0, min(1.0, x1p / 100.0)) * w)
        y0 = int(max(0.0, min(1.0, y0p / 100.0)) * h)
        y1 = int(max(0.0, min(1.0, y1p / 100.0)) * h)
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, label, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    _box(int(st.session_state.get("roi_esq_x0", DEFAULT_ROI["ESQ"]["x0"])), int(st.session_state.get("roi_esq_x1", DEFAULT_ROI["ESQ"]["x1"])), int(st.session_state.get("roi_esq_y0", DEFAULT_ROI["ESQ"]["y0"])), int(st.session_state.get("roi_esq_y1", DEFAULT_ROI["ESQ"]["y1"])), (0, 200, 0), "ROI ESQ")
    _box(int(st.session_state.get("roi_dir_x0", DEFAULT_ROI["DIR"]["x0"])), int(st.session_state.get("roi_dir_x1", DEFAULT_ROI["DIR"]["x1"])), int(st.session_state.get("roi_dir_y0", DEFAULT_ROI["DIR"]["y0"])), int(st.session_state.get("roi_dir_y1", DEFAULT_ROI["DIR"]["y1"])), (0, 140, 255), "ROI DIR")
    return img

def append_evidence_audit_csv(row: dict):
    fieldnames = [
        "timestamp", "inspection_id", "source", "saved_class", "detail_code", "final_result",
        "attention_flag", "confidence_proxy", "p_pres_esq", "p_pres_dir",
        "prob_ng_esq", "prob_ng_dir", "decision_band_esq", "decision_band_dir",
        "production_order", "serial_number", "model_name", "image_path",
        "overlay_path", "metadata_path", "reason",
    ]
    file_exists = AUDIT_LOG_PATH.exists()
    with open(AUDIT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

def build_evidence_bundle(result: dict | None = None, frame_bgr: np.ndarray | None = None) -> dict:
    result = result or st.session_state.get("last_result") or {}
    frame_bgr = frame_bgr if frame_bgr is not None else st.session_state.get("display_frame")
    return {
        "frame_bgr": frame_bgr.copy() if isinstance(frame_bgr, np.ndarray) else None,
        "overlay_bgr": draw_roi_overlay(frame_bgr) if isinstance(frame_bgr, np.ndarray) else None,
        "roi_esq": result.get("roi_esq"),
        "roi_dir": result.get("roi_dir"),
        "result": result,
    }

def save_evidence_bundle(save_root: Path, label: str, reason: str, source: str, result: dict | None = None, frame_bgr: np.ndarray | None = None) -> dict:
    bundle = build_evidence_bundle(result=result, frame_bgr=frame_bgr)
    result = bundle["result"] or {}
    frame_bgr = bundle["frame_bgr"]
    overlay_bgr = bundle["overlay_bgr"]
    roi_esq = bundle["roi_esq"]
    roi_dir = bundle["roi_dir"]

    save_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    inspection_id = str(st.session_state.get("last_inspection_id", "") or generate_inspection_id())
    serial_number = normalize_serial_qr(st.session_state.get("serial_qr_code", ""))
    serial_tag = sanitize_filename(serial_number) if serial_number else "SEM_SERIAL"
    prefix = f"{stamp}__{label}__{serial_tag}"

    raw_path = save_root / f"{prefix}__raw.jpg"
    overlay_path = save_root / f"{prefix}__overlay.jpg"
    roi_esq_path = save_root / f"{prefix}__roi_esq.jpg"
    roi_dir_path = save_root / f"{prefix}__roi_dir.jpg"
    meta_path = save_root / f"{prefix}__meta.json"

    if isinstance(frame_bgr, np.ndarray):
        cv2.imwrite(str(raw_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if isinstance(overlay_bgr, np.ndarray):
        cv2.imwrite(str(overlay_path), overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if isinstance(roi_esq, np.ndarray):
        cv2.imwrite(str(roi_esq_path), roi_esq, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if isinstance(roi_dir, np.ndarray):
        cv2.imwrite(str(roi_dir_path), roi_dir, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    confidence_proxy = float(max(result.get("prob_ng_esq", 0.0), result.get("prob_ng_dir", 0.0), 0.0))
    detail_code = build_defect_detail_code(result)

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inspection_id": inspection_id,
        "source": source,
        "saved_class": label,
        "saved_detail": detail_code,
        "reason": reason,
        "production_order": str(st.session_state.get("production_order", "")).strip(),
        "serial_number": serial_number,
        "model_name": str(st.session_state.get("product_model", "")),
        "equipment_id": str(st.session_state.get("equipment_id", "")).strip(),
        "user_mode": str(st.session_state.get("user_mode", "OPERADOR")),
        "result": {
            "final_result": result.get("defect_type", ""),
            "defect_esq": result.get("defect_esq", ""),
            "defect_dir": result.get("defect_dir", ""),
            "attention_flag": bool(result.get("attention_flag", False)),
            "decision_band_esq": result.get("decision_band_esq", ""),
            "decision_band_dir": result.get("decision_band_dir", ""),
            "p_pres_esq": float(result.get("p_pres_esq", 0.0)),
            "p_pres_dir": float(result.get("p_pres_dir", 0.0)),
            "prob_ng_esq": float(result.get("prob_ng_esq", 0.0)),
            "prob_ng_dir": float(result.get("prob_ng_dir", 0.0)),
            "prob_ok_esq": float(result.get("prob_ok_esq", 0.0)),
            "prob_ok_dir": float(result.get("prob_ok_dir", 0.0)),
            "confidence_proxy": confidence_proxy,
            "threshold_presente": float(st.session_state.get("threshold_presente", DEFAULT_THRESH_PRESENTE)),
            "threshold_ng_ok": float(st.session_state.get("threshold_ng_ok", DEFAULT_THR_NG_OK)),
            "threshold_ng_ng": float(st.session_state.get("threshold_ng_ng", DEFAULT_THR_NG_NG)),
        },
        "files": {
            "raw": raw_path.name if raw_path.exists() else "",
            "overlay": overlay_path.name if overlay_path.exists() else "",
            "roi_esq": roi_esq_path.name if roi_esq_path.exists() else "",
            "roi_dir": roi_dir_path.name if roi_dir_path.exists() else "",
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    append_evidence_audit_csv({
        "timestamp": metadata["timestamp"],
        "inspection_id": inspection_id,
        "source": source,
        "saved_class": label,
        "detail_code": detail_code,
        "final_result": result.get("defect_type", ""),
        "attention_flag": bool(result.get("attention_flag", False)),
        "confidence_proxy": f"{confidence_proxy:.6f}",
        "p_pres_esq": f"{float(result.get('p_pres_esq', 0.0)):.6f}",
        "p_pres_dir": f"{float(result.get('p_pres_dir', 0.0)):.6f}",
        "prob_ng_esq": f"{float(result.get('prob_ng_esq', 0.0)):.6f}",
        "prob_ng_dir": f"{float(result.get('prob_ng_dir', 0.0)):.6f}",
        "decision_band_esq": result.get("decision_band_esq", ""),
        "decision_band_dir": result.get("decision_band_dir", ""),
        "production_order": metadata["production_order"],
        "serial_number": serial_number,
        "model_name": metadata["model_name"],
        "image_path": str(raw_path),
        "overlay_path": str(overlay_path),
        "metadata_path": str(meta_path),
        "reason": reason,
    })

    return {
        "raw_path": raw_path,
        "overlay_path": overlay_path,
        "roi_esq_path": roi_esq_path,
        "roi_dir_path": roi_dir_path,
        "meta_path": meta_path,
        "inspection_id": inspection_id,
        "reason": reason,
        "label": label,
    }

def cleanup_old_evidence(base_folder: Path, retention_days: int) -> tuple[int, int]:
    deleted_files = 0
    deleted_bytes = 0
    if not base_folder.exists():
        return deleted_files, deleted_bytes
    cutoff = datetime.now() - timedelta(days=int(retention_days))
    for p in base_folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".json"):
            continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            if mtime < cutoff:
                size = p.stat().st_size
                p.unlink(missing_ok=True)
                deleted_files += 1
                deleted_bytes += size
        except Exception:
            pass
    return deleted_files, deleted_bytes

def maybe_cleanup_auto_evidence(base_folder: Path, retention_days: int, enabled: bool, interval_sec: int = 1800) -> tuple[int, int]:
    if not enabled:
        return 0, 0
    now_ts = time.time()
    last_run = float(st.session_state.get("evidence_last_cleanup_ts", 0.0))
    if (now_ts - last_run) < float(interval_sec):
        return 0, 0
    deleted_files, deleted_bytes = cleanup_old_evidence(base_folder, retention_days)
    st.session_state["evidence_last_cleanup_ts"] = now_ts
    st.session_state["evidence_last_cleanup_files"] = deleted_files
    st.session_state["evidence_last_cleanup_bytes"] = deleted_bytes
    return deleted_files, deleted_bytes

def should_auto_save_evidence(res: dict) -> tuple[bool, str | None, str | None]:
    final_label = map_defect_label(res.get("defect_type", ""))
    if final_label == "NG_DESALINHADO":
        return True, "NG_DESALINHADO", "Falha automática: NG_DESALINHADO"
    if final_label == "NG_FALTANDO":
        return True, "NG_FALTANDO", "Falha automática: NG_FALTANDO"
    save_ok_limit = bool(st.session_state.get("evidence_save_ok_limit", True))
    if final_label == "OK" and save_ok_limit and bool(res.get("attention_flag", False)):
        return True, "OK_LIMITE", "Aprovado próximo do limite / banda de atenção"
    return False, None, None

def auto_save_current_result_if_needed(res: dict, frame_bgr: np.ndarray | None = None, source: str = "auto") -> dict | None:
    if not bool(st.session_state.get("evidence_auto_enabled", True)):
        return None
    should_save, label, reason = should_auto_save_evidence(res)
    if not should_save or not label:
        return None
    current_signature = f"{res.get('defect_type','')}_{float(res.get('prob_ng_esq',0.0)):.4f}_{float(res.get('prob_ng_dir',0.0)):.4f}_{float(res.get('p_pres_esq',0.0)):.4f}_{float(res.get('p_pres_dir',0.0)):.4f}"
    last_signature = st.session_state.get("evidence_last_auto_signature")
    last_ts = float(st.session_state.get("evidence_last_auto_ts", 0.0))
    now_ts = time.time()
    if current_signature == last_signature and (now_ts - last_ts) < 1.2:
        return None
    detail_code = build_defect_detail_code(res)
    target_dir = AUTO_EVIDENCE_CLASSES[label] / detail_code
    saved = save_evidence_bundle(target_dir, label, reason, source, result=res, frame_bgr=frame_bgr)
    st.session_state["evidence_last_auto_signature"] = current_signature
    st.session_state["evidence_last_auto_ts"] = now_ts
    st.session_state["evidence_last_saved"] = saved
    return saved

def save_manual_current_result(label: str | None = None, detail_override: str | None = None) -> dict:
    result = st.session_state.get("last_result") or {}
    detail_code = str(detail_override or build_defect_detail_code(result)).strip().upper()
    final_label = str(label or manual_label_from_detail(detail_code)).strip().upper()
    folder = MANUAL_EVIDENCE_CLASSES[final_label] / detail_code
    reason = f"Salvamento manual pelo operador/engenharia: {final_label} / {detail_code}"
    saved = save_evidence_bundle(folder, final_label, reason, "manual", result=result, frame_bgr=st.session_state.get("display_frame"))
    st.session_state["evidence_last_manual"] = saved
    st.session_state["manual_last_saved_detail"] = detail_code
    inc_manual_detail_counter(detail_code)
    return saved

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


def read_one_frame_timeout(cap, timeout_s: float = 1.5):
    """Lê 1 frame com timeout para evitar travar a UI (drivers podem bloquear em cap.read)."""
    out = {"ok": False, "frame": None, "err": None}

    def _worker():
        try:
            ok, frame = cap.read()
            out["ok"] = bool(ok)
            out["frame"] = frame
        except Exception as e:
            out["err"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        raise TimeoutError(f"Timeout em cap.read() após {timeout_s}s")

    if out["err"] is not None:
        raise out["err"]

    if (not out["ok"]) or (out["frame"] is None):
        return None

    return out["frame"]




def decide_misaligned_status(prob_ng: float, prob_ok: float, thr_ng_ok: float, thr_ng_ng: float, margin_abs: float = 0.10):
    """Decisão industrial para desalinhamento com banda de atenção."""
    prob_ng = float(prob_ng)
    prob_ok = float(prob_ok)
    thr_ng_ok = float(thr_ng_ok)
    thr_ng_ng = float(thr_ng_ng)
    margin_abs = float(margin_abs)

    margin = prob_ng - prob_ok

    if prob_ng >= thr_ng_ng and margin >= margin_abs:
        return "NG_MISALIGNED", "NG_STRONG", margin, False
    if prob_ng >= thr_ng_ng and margin < margin_abs:
        return "OK", "ATTENTION", margin, True
    if prob_ng >= thr_ng_ok:
        return "OK", "ATTENTION", margin, True
    return "OK", "OK_SAFE", margin, False


def infer_dual_on_frame(frame_bgr: np.ndarray):
    """Inferência DUAL com lógica industrial calibrada para chão de fábrica.

    Saídas industriais:
      - OK
      - NG_MISSING
      - NG_MISALIGNED
    """

    esq_x0 = int(st.session_state.get("roi_esq_x0", DEFAULT_ROI["ESQ"]["x0"]))
    esq_x1 = int(st.session_state.get("roi_esq_x1", DEFAULT_ROI["ESQ"]["x1"]))
    esq_y0 = int(st.session_state.get("roi_esq_y0", DEFAULT_ROI["ESQ"]["y0"]))
    esq_y1 = int(st.session_state.get("roi_esq_y1", DEFAULT_ROI["ESQ"]["y1"]))

    dir_x0 = int(st.session_state.get("roi_dir_x0", DEFAULT_ROI["DIR"]["x0"]))
    dir_x1 = int(st.session_state.get("roi_dir_x1", DEFAULT_ROI["DIR"]["x1"]))
    dir_y0 = int(st.session_state.get("roi_dir_y0", DEFAULT_ROI["DIR"]["y0"]))
    dir_y1 = int(st.session_state.get("roi_dir_y1", DEFAULT_ROI["DIR"]["y1"]))

    normalize_roi = bool(st.session_state.get("normalize_lab_equalize", DEFAULT_NORMALIZE_LAB))

    roi_esq = crop_roi_percent(frame_bgr, esq_x0, esq_x1, esq_y0, esq_y1)
    roi_dir = crop_roi_percent(frame_bgr, dir_x0, dir_x1, dir_y0, dir_y1)

    if normalize_roi:
        roi_esq = equalize_lab_bgr(roi_esq)
        roi_dir = equalize_lab_bgr(roi_dir)

    th_presente = float(st.session_state.get("threshold_presente", DEFAULT_THRESH_PRESENTE))

    p_pres_esq = 1.0
    p_pres_dir = 1.0
    cls_pres_esq = "mola_presente"
    cls_pres_dir = "mola_presente"
    probs_pres_esq = None
    probs_pres_dir = None

    try:
        ensure_model_loaded_or_raise(blocking=True)
        if st.session_state.get("model") is not None and st.session_state.get("labels") is not None:
            cls_pres_esq, conf_esq_, probs_pres_esq = predict_one(
                st.session_state["model"], st.session_state["labels"], roi_esq
            )
            cls_pres_dir, conf_dir_, probs_pres_dir = predict_one(
                st.session_state["model"], st.session_state["labels"], roi_dir
            )
            p_pres_esq = prob_of_class(st.session_state["labels"], probs_pres_esq, "mola_presente")
            p_pres_dir = prob_of_class(st.session_state["labels"], probs_pres_dir, "mola_presente")
    except Exception:
        pass

    missing_esq = (p_pres_esq < th_presente)
    missing_dir = (p_pres_dir < th_presente)

    use_mnv2 = bool(st.session_state.get("use_mnv2_prod", True))

    cls_mis_esq = "OK"
    cls_mis_dir = "OK"
    prob_ng_esq = 0.0
    prob_ng_dir = 0.0
    prob_ok_esq = 1.0
    prob_ok_dir = 1.0
    probs_mis_esq = None
    probs_mis_dir = None

    thr_ng_ok = float(st.session_state.get("threshold_ng_ok", DEFAULT_THR_NG_OK))
    thr_ng_ng = float(st.session_state.get("threshold_ng_ng", DEFAULT_THR_NG_NG))
    margin_abs = float(st.session_state.get("prod_margin_abs", 0.10))

    decision_band_esq = "MISSING" if missing_esq else "OK_SAFE"
    decision_band_dir = "MISSING" if missing_dir else "OK_SAFE"
    margin_esq = 1.0
    margin_dir = 1.0
    attention_esq = False
    attention_dir = False

    if use_mnv2:
        ensure_prod_model_loaded_or_raise(blocking=True)
        model = st.session_state.get("prod_model")
        class_names = st.session_state.get("prod_class_names")
        pos_idx = int(st.session_state.get("prod_pos_idx", 0))
        img_size = tuple(st.session_state.get("prod_img_size", (224, 224)))

        if model is None or class_names is None:
            raise RuntimeError("Modelo PRODUÇÃO não carregado.")

        if not missing_esq:
            _, prob_ng_esq, probs_mis_esq = infer_mobilenetv2_prod(
                roi_esq, model, class_names, pos_idx, thr_ng_ng, img_size=img_size
            )
            prob_ok_esq = float((probs_mis_esq or {}).get("OK", 1.0 - prob_ng_esq))
            cls_mis_esq, decision_band_esq, margin_esq, attention_esq = decide_misaligned_status(
                prob_ng_esq, prob_ok_esq, thr_ng_ok, thr_ng_ng, margin_abs=margin_abs
            )
        else:
            cls_mis_esq, prob_ng_esq, probs_mis_esq = ("OK", 0.0, None)

        if not missing_dir:
            _, prob_ng_dir, probs_mis_dir = infer_mobilenetv2_prod(
                roi_dir, model, class_names, pos_idx, thr_ng_ng, img_size=img_size
            )
            prob_ok_dir = float((probs_mis_dir or {}).get("OK", 1.0 - prob_ng_dir))
            cls_mis_dir, decision_band_dir, margin_dir, attention_dir = decide_misaligned_status(
                prob_ng_dir, prob_ok_dir, thr_ng_ok, thr_ng_ng, margin_abs=margin_abs
            )
        else:
            cls_mis_dir, prob_ng_dir, probs_mis_dir = ("OK", 0.0, None)

    mis_esq = (cls_mis_esq == "NG_MISALIGNED") and (not missing_esq)
    mis_dir = (cls_mis_dir == "NG_MISALIGNED") and (not missing_dir)

    defect_esq = "OK"
    defect_dir = "OK"

    if missing_esq:
        defect_esq = "NG_MISSING"
    elif mis_esq:
        defect_esq = "NG_MISALIGNED"

    if missing_dir:
        defect_dir = "NG_MISSING"
    elif mis_dir:
        defect_dir = "NG_MISALIGNED"

    if defect_esq == "NG_MISSING" or defect_dir == "NG_MISSING":
        defect_type = "NG_MISSING"
    elif defect_esq == "NG_MISALIGNED" or defect_dir == "NG_MISALIGNED":
        defect_type = "NG_MISALIGNED"
    else:
        defect_type = "OK"

    attention_flag = bool(attention_esq or attention_dir)
    aprovado = (defect_type == "OK")
    ok_esq = (defect_esq == "OK")
    ok_dir = (defect_dir == "OK")

    return {
        "roi_esq": roi_esq,
        "roi_dir": roi_dir,
        "cls_esq": cls_pres_esq,
        "cls_dir": cls_pres_dir,
        "p_pres_esq": float(p_pres_esq),
        "p_pres_dir": float(p_pres_dir),
        "conf_esq": float(p_pres_esq),
        "conf_dir": float(p_pres_dir),
        "cls_mnv2_esq": cls_mis_esq,
        "cls_mnv2_dir": cls_mis_dir,
        "prob_ng_esq": float(prob_ng_esq),
        "prob_ng_dir": float(prob_ng_dir),
        "prob_ok_esq": float(prob_ok_esq),
        "prob_ok_dir": float(prob_ok_dir),
        "thr_ng": float(thr_ng_ng),
        "thr_ng_ok": float(thr_ng_ok),
        "thr_ng_ng": float(thr_ng_ng),
        "margin_esq": float(margin_esq),
        "margin_dir": float(margin_dir),
        "decision_band_esq": decision_band_esq,
        "decision_band_dir": decision_band_dir,
        "attention_esq": bool(attention_esq),
        "attention_dir": bool(attention_dir),
        "attention_flag": attention_flag,
        "probs_esq": probs_mis_esq,
        "probs_dir": probs_mis_dir,
        "defect_esq": defect_esq,
        "defect_dir": defect_dir,
        "defect_type": defect_type,
        "ok_esq": ok_esq,
        "ok_dir": ok_dir,
        "aprovado": aprovado,
    }

def infer_dual_on_frame_timeout(frame_bgr, timeout_s: float = 4.0):
    """Executa a inferência com timeout (protege contra travamentos raros do backend/TF)."""
    out = {"res": None, "err": None}

    def _worker():
        try:
            out["res"] = infer_dual_on_frame(frame_bgr)
        except Exception as e:
            out["err"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        raise TimeoutError(f"Timeout em infer_dual_on_frame() após {timeout_s}s")

    if out["err"] is not None:
        raise out["err"]

    return out["res"]

def quick_frame_signature(frame_bgr: np.ndarray, size: int = 48) -> np.ndarray:
    """Assinatura rápida do frame para detectar troca de peça (sem depender do sensor voltar a 0).
    Retorna um array uint8 (thumbnail em tons de cinza) para comparar diferenças.
    """
    try:
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        return g.astype(np.uint8)
    except Exception:
        # fallback
        return np.zeros((size, size), dtype=np.uint8)

def signature_diff(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 9999.0
    if a.shape != b.shape:
        return 9999.0
    return float(np.mean(cv2.absdiff(a, b)))



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

def ensure_model_loaded_or_raise(blocking: bool = False):
    """Legacy loader (modelo_molas.keras + labels.json / registry)."""
    st.session_state["model_loading"] = False
    if st.session_state.get("model") is not None and st.session_state.get("labels") is not None:
        return

    acquired = MODEL_LOAD_LOCK.acquire(blocking=blocking)
    if not acquired:
        # Sensor / rerun rápido: não explode; apenas sinaliza e tenta no próximo ciclo
        st.session_state["model_loading"] = True
        return

    try:
        paths = st.session_state.get("selected_model_paths")
        if paths and len(paths) == 3:
            model_p = Path(paths[0])
            labels_p = Path(paths[1])
        else:
            model_p = MODEL_PATH
            labels_p = LABELS_PATH

        st.session_state["labels"] = load_labels(labels_p)
        st.session_state["model"] = load_model_cached(str(model_p))
    finally:
        MODEL_LOAD_LOCK.release()


def ensure_prod_model_loaded_or_raise(blocking: bool = False):
    """Loader do MobileNetV2 de produção (model_final.keras + production_package.json)."""
    if not st.session_state.get("use_mnv2_prod", True):
        return

    if st.session_state.get("prod_model") is not None and st.session_state.get("prod_class_names") is not None:
        return

    acquired = MODEL_LOAD_LOCK.acquire(blocking=blocking)
    if not acquired:
        st.session_state["model_loading"] = True
        return

    try:
        # Valida pasta
        if not PROD_MODEL_DIR.exists():
            raise FileNotFoundError(f"Pasta do modelo produção não encontrada: {PROD_MODEL_DIR}")

        class_names, pos_name, pos_idx, thr_ng, img_size = load_production_package(str(PROD_MODEL_DIR))
        model, model_path = load_mobilenetv2_prod_model(str(PROD_MODEL_DIR))

        st.session_state["prod_model"] = model
        st.session_state["prod_model_path"] = model_path
        st.session_state["prod_class_names"] = class_names
        st.session_state["prod_pos_idx"] = int(pos_idx)
        st.session_state["prod_thr_ng"] = float(thr_ng)
        st.session_state["prod_img_size"] = tuple(img_size)

        # Defaults industriais conservadores para campo.
        st.session_state.setdefault("threshold_presente", DEFAULT_THRESH_PRESENTE)
        st.session_state.setdefault("threshold_ng_ok", max(0.0, float(thr_ng) + 0.05))
        st.session_state.setdefault("threshold_ng_ng", max(float(st.session_state.get("threshold_ng_ok", DEFAULT_THR_NG_OK)) + 0.10, float(thr_ng) + 0.20))
    finally:
        MODEL_LOAD_LOCK.release()


def ensure_active_model_loaded_or_raise(blocking: bool = False):
    """Garante que o modelo ativo (produção ou legacy) esteja carregado."""
    if st.session_state.get("use_mnv2_prod", True):
        ensure_prod_model_loaded_or_raise(blocking=blocking)
        # sanity
        if st.session_state.get("prod_model") is None:
            raise RuntimeError("Modelo PRODUÇÃO não carregado.")
    else:
        ensure_model_loaded_or_raise(blocking=blocking)
        if st.session_state.get("model") is None or st.session_state.get("labels") is None:
            raise RuntimeError("Modelo LEGACY não carregado.")

def decode_uploaded_image_to_bgr(uploaded_file) -> np.ndarray:
    """Decodifica UploadedFile/bytes em imagem BGR para inferência offline."""
    if uploaded_file is None:
        raise ValueError("Nenhum arquivo enviado.")

    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if not data:
        raise ValueError("Arquivo de imagem vazio.")

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem enviada.")
    return img


def run_infer_dual_on_uploaded_frame(frame_bgr: np.ndarray, file_name: str = "upload", update_metrics: bool = False):
    """Executa a inferência DUAL em uma imagem enviada manualmente (sem câmera/sensor)."""
    try:
        ensure_active_model_loaded_or_raise(blocking=True)
    except Exception as e:
        st.session_state["last_error"] = f"Falha ao carregar modelo atual: {e}"
        st.session_state["last_result"] = None
        return None

    if frame_bgr is None or getattr(frame_bgr, 'size', 0) == 0:
        st.session_state["last_error"] = "Imagem enviada inválida para inferência."
        st.session_state["last_result"] = None
        return None

    src = frame_bgr.copy()
    st.session_state["display_frame"] = src.copy()
    st.session_state["last_frame"] = src.copy()
    st.session_state["upload_test_frame"] = src.copy()
    st.session_state["upload_test_name"] = str(file_name or "upload")
    st.session_state["last_infer_sig"] = quick_frame_signature(src)
    st.session_state["last_infer_ts"] = time.time()

    start_dt = datetime.now()
    try:
        res = infer_dual_on_frame(src)
        end_dt = datetime.now()
        st.session_state["last_result"] = res
        st.session_state["last_error"] = None

        if bool(update_metrics):
            update_metrics_and_history(res)

        st.session_state["last_warning"] = f"Inspeção por upload executada: {file_name}"
        return res
    except Exception as e:
        st.session_state["last_error"] = f"Falha na inferência por upload: {e}"
        st.session_state["last_result"] = None
        return None



def update_metrics_and_history(res: dict) -> None:
    """Atualiza contadores de produção e histórico (shim antecipado para upload)."""
    st.session_state["cnt_total"] = int(st.session_state.get("cnt_total", 0)) + 1

    aprovado = bool(res.get("aprovado", False))
    if aprovado:
        st.session_state["cnt_ok"] = int(st.session_state.get("cnt_ok", 0)) + 1
        if bool(res.get("attention_flag", False)):
            st.session_state["cnt_ok_attention"] = int(st.session_state.get("cnt_ok_attention", 0)) + 1
    else:
        st.session_state["cnt_ng"] = int(st.session_state.get("cnt_ng", 0)) + 1
        if not bool(res.get("ok_esq", True)):
            st.session_state["cnt_ng_esq"] = int(st.session_state.get("cnt_ng_esq", 0)) + 1
        if not bool(res.get("ok_dir", True)):
            st.session_state["cnt_ng_dir"] = int(st.session_state.get("cnt_ng_dir", 0)) + 1

        defect_esq = str(res.get("defect_esq", "OK") or "OK").strip().upper()
        defect_dir = str(res.get("defect_dir", "OK") or "OK").strip().upper()

        if defect_esq == "NG_MISSING" and defect_dir == "OK":
            st.session_state["cnt_missing_esq"] = int(st.session_state.get("cnt_missing_esq", 0)) + 1
        elif defect_esq == "OK" and defect_dir == "NG_MISSING":
            st.session_state["cnt_missing_dir"] = int(st.session_state.get("cnt_missing_dir", 0)) + 1
        elif defect_esq == "NG_MISSING" and defect_dir == "NG_MISSING":
            st.session_state["cnt_missing_both"] = int(st.session_state.get("cnt_missing_both", 0)) + 1
        elif defect_esq == "NG_MISALIGNED" and defect_dir == "OK":
            st.session_state["cnt_misaligned_esq"] = int(st.session_state.get("cnt_misaligned_esq", 0)) + 1
        elif defect_esq == "OK" and defect_dir == "NG_MISALIGNED":
            st.session_state["cnt_misaligned_dir"] = int(st.session_state.get("cnt_misaligned_dir", 0)) + 1
        elif defect_esq == "NG_MISALIGNED" and defect_dir == "NG_MISALIGNED":
            st.session_state["cnt_misaligned_both"] = int(st.session_state.get("cnt_misaligned_both", 0)) + 1
        else:
            st.session_state["cnt_misto"] = int(st.session_state.get("cnt_misto", 0)) + 1

    history = list(st.session_state.get("history", []))
    history.append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "total": int(st.session_state.get("cnt_total", 0)),
        "ok": int(st.session_state.get("cnt_ok", 0)),
        "ng": int(st.session_state.get("cnt_ng", 0)),
        "yield_pct": (float(st.session_state.get("cnt_ok", 0)) / max(1, float(st.session_state.get("cnt_total", 1)))) * 100.0,
        "result": "OK" if aprovado else str(res.get("defect_type", "NG")),
    })
    st.session_state["history"] = history[-500:]

# ==========================================================
# LOG CSV
# ==========================================================
def append_log_csv(row: dict):
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"inspecao_molas_{today}.csv"

    fieldnames = [
        "timestamp", "modelo", "linha", "resultado_final",
        "defect_esq", "defect_dir",
        "cs_code", "cs_detail", "p_esq", "p_dir", "th_presente",
        "camera_index", "directshow", "source",
        "total", "ok", "ng", "yield_pct",
        "test_time_sec", "start_time", "end_time",
    ]

    file_exists = log_path.exists()

    # Se o arquivo já existe e o header antigo não tem as novas colunas,
    # reescreve preservando os dados (para manter compatibilidade).
    if file_exists:
        try:
            with open(log_path, "r", newline="", encoding="utf-8") as rf:
                reader = csv.reader(rf, delimiter=";")
                existing_header = next(reader, [])
        except Exception:
            existing_header = []

        if existing_header and existing_header != fieldnames:
            tmp_path = log_path.with_suffix(".tmp")
            try:
                with open(log_path, "r", newline="", encoding="utf-8") as rf, open(tmp_path, "w", newline="", encoding="utf-8") as wf:
                    dr = csv.DictReader(rf, delimiter=";")
                    dw = csv.DictWriter(wf, fieldnames=fieldnames, delimiter=";")
                    dw.writeheader()
                    for old_row in dr:
                        dw.writerow({k: old_row.get(k, "") for k in fieldnames})
                os.replace(tmp_path, log_path)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

def get_cs_code(res: dict, th: float) -> tuple[str, str]:
    """Código curto para UI/CSV.

    Padrão industrial:
      - se res['defect_type'] existir: OK | NG_MISSING | NG_MISALIGNED
      - senão, mantém o fallback antigo (OK / NG_ESQ / NG_DIR / NG_AMBAS)
    """
    # Novo padrão industrial
    defect_type = str(res.get("defect_type", "")).strip().upper()
    if defect_type in ("OK", "NG_MISSING", "NG_MISALIGNED"):
        d_esq = str(res.get("defect_esq", ""))
        d_dir = str(res.get("defect_dir", ""))
        detail = f"ESQ={d_esq} | DIR={d_dir} | th_pres={th:.2f}"
        return defect_type, detail

    # Fallback antigo
    ok_esq = res.get("ok_esq", False)
    ok_dir = res.get("ok_dir", False)

    if ok_esq and ok_dir:
        return "OK", "OK"
    if (not ok_esq) and ok_dir:
        return "NG_ESQ", f"p_esq<{th:.2f}"
    if ok_esq and (not ok_dir):
        return "NG_DIR", f"p_dir<{th:.2f}"
    return "NG_AMBAS", f"p_esq<{th:.2f} | p_dir<{th:.2f}"



def report_summary_snapshot() -> dict:
    total = int(st.session_state.get("cnt_total", 0))
    ok = int(st.session_state.get("cnt_ok", 0))
    ng = int(st.session_state.get("cnt_ng", 0))
    yield_pct = (ok / total * 100.0) if total > 0 else 0.0
    audit = get_audit_counts_from_session()
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "period_start": str(st.session_state.get("production_started_at", "---")),
        "line_name": str(st.session_state.get("line_name", "L01")),
        "equipment_id": str(st.session_state.get("equipment_id", "SVC01")),
        "model_name": str(st.session_state.get("selected_model_key", "MODELO_PADRAO")),
        "production_order": str(st.session_state.get("production_order", "")).strip() or "---",
        "inspection_id": str(st.session_state.get("last_inspection_id", "")).strip() or "---",
        "total": total,
        "ok": ok,
        "ng": ng,
        "yield_pct": yield_pct,
        "audit": audit,
    }


def _save_report_charts(snapshot: dict, stamp: str) -> list[Path]:
    charts = []
    if not HAS_MPL:
        return charts
    import matplotlib.pyplot as plt

    # Chart 1: OK vs NG
    fig = plt.figure(figsize=(6, 4))
    plt.bar(["OK", "NG"], [snapshot["ok"], snapshot["ng"]])
    plt.ylabel("Quantidade")
    plt.title("Resumo de Produção")
    p1 = REPORTS_DIR / f"audit_report_{stamp}_ok_ng.png"
    fig.savefig(p1, bbox_inches="tight", dpi=180)
    plt.close(fig)
    charts.append(p1)

    # Chart 2: tipos de falha
    audit = snapshot["audit"]
    labels = [
        "Falt. ESQ", "Falt. DIR", "Falt. BOTH",
        "Des. ESQ", "Des. DIR", "Des. BOTH",
        "Misto", "OK atenção"
    ]
    values = [
        audit["faltando_esq"], audit["faltando_dir"], audit["faltando_both"],
        audit["desalinhada_esq"], audit["desalinhada_dir"], audit["desalinhada_both"],
        audit["misto"], audit["ok_atencao"],
    ]
    fig = plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.ylabel("Ocorrências")
    plt.title("Detalhamento de Falhas")
    plt.xticks(rotation=25, ha='right')
    p2 = REPORTS_DIR / f"audit_report_{stamp}_falhas.png"
    fig.savefig(p2, bbox_inches="tight", dpi=180)
    plt.close(fig)
    charts.append(p2)
    return charts


def _build_report_html(snapshot: dict) -> str:
    audit = snapshot["audit"]
    return f"""<!DOCTYPE html>
<html lang="pt-br"><head><meta charset="utf-8"><title>Resumo Relatório de Auditoria</title></head>
<body style="font-family:Arial,sans-serif;font-size:14px;color:#111827;">
  <h2 style="margin-bottom:8px;">Resumo do Relatório de Auditoria – SVC Inspeção de Molas</h2>
  <p><b>Emitido em:</b> {snapshot['generated_at']}<br>
     <b>Período:</b> {snapshot['period_start']} até {snapshot['generated_at']}<br>
     <b>Linha:</b> {snapshot['line_name']} &nbsp; | &nbsp; <b>Equipamento:</b> {snapshot['equipment_id']}<br>
     <b>Modelo:</b> {snapshot['model_name']} &nbsp; | &nbsp; <b>OP:</b> {snapshot['production_order']}</p>
  <table style="border-collapse:collapse;margin:8px 0;" border="1" cellpadding="6">
    <tr><th>Total</th><th>OK</th><th>NG</th><th>Yield</th></tr>
    <tr><td>{snapshot['total']}</td><td>{snapshot['ok']}</td><td>{snapshot['ng']}</td><td>{snapshot['yield_pct']:.2f}%</td></tr>
  </table>
  <p><b>Falhas:</b><br>
     Faltando ESQ: {audit['faltando_esq']}<br>
     Faltando DIR: {audit['faltando_dir']}<br>
     Faltando BOTH: {audit['faltando_both']}<br>
     Desalinhada ESQ: {audit['desalinhada_esq']}<br>
     Desalinhada DIR: {audit['desalinhada_dir']}<br>
     Desalinhada BOTH: {audit['desalinhada_both']}<br>
     Casos mistos: {audit['misto']}<br>
     OK com atenção: {audit['ok_atencao']}</p>
  <p>Relatório completo em PDF anexo.</p>
</body></html>"""


def generate_audit_report_files() -> tuple[Path, Path]:
    snapshot = report_summary_snapshot()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    charts = _save_report_charts(snapshot, stamp)

    html_content = _build_report_html(snapshot)
    html_path = REPORTS_DIR / f"audit_report_{stamp}.html"
    html_path.write_text(html_content, encoding='utf-8')

    pdf_path = REPORTS_DIR / f"audit_report_{stamp}.pdf"

    if not HAS_MPL:
        raise RuntimeError("matplotlib não está disponível para gerar o PDF do relatório.")

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap

    audit = snapshot['audit']

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        y = 0.965
        def line(txt, size=10, weight='normal', color='#111827', step=0.026):
            nonlocal y
            fig.text(0.07, y, txt, fontsize=size, fontweight=weight, color=color, va='top', ha='left')
            y -= step

        line('SVC Inspeção de Molas — DUAL', size=18, weight='bold', step=0.035)
        line('Relatório de Auditoria', size=14, weight='bold', step=0.032)
        line(f"Emitido em: {snapshot['generated_at']}", size=10)
        line(f"Período: {snapshot['period_start']} até {snapshot['generated_at']}", size=10)
        line(f"Linha: {snapshot['line_name']}    Equipamento: {snapshot['equipment_id']}", size=10)
        line(f"Modelo: {snapshot['model_name']}    OP: {snapshot['production_order']}", size=10)
        line(f"Último Inspection ID: {snapshot['inspection_id']}", size=10, step=0.034)

        line('Resumo da produção', size=12, weight='bold', step=0.03)
        line(f"Total produzido: {snapshot['total']}", size=10)
        line(f"Total OK: {snapshot['ok']}", size=10)
        line(f"Total NG: {snapshot['ng']}", size=10)
        line(f"Yield: {snapshot['yield_pct']:.2f}%", size=10, step=0.034)

        line('Detalhamento das falhas', size=12, weight='bold', step=0.03)
        details = [
            f"Faltando ESQ: {audit['faltando_esq']}",
            f"Faltando DIR: {audit['faltando_dir']}",
            f"Faltando BOTH: {audit['faltando_both']}",
            f"Desalinhada ESQ: {audit['desalinhada_esq']}",
            f"Desalinhada DIR: {audit['desalinhada_dir']}",
            f"Desalinhada BOTH: {audit['desalinhada_both']}",
            f"Casos mistos: {audit['misto']}",
            f"OK com atenção: {audit['ok_atencao']}",
        ]
        for item in details:
            line(item, size=10)

        y -= 0.01
        wrapped = textwrap.fill(
            'Relatório gerado automaticamente pelo sistema com base nos dados disponíveis até o momento da emissão.',
            width=95
        )
        fig.text(0.07, max(y, 0.06), wrapped, fontsize=9, color='#4b5563', va='top', ha='left')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        for cp in charts:
            if cp.exists():
                img = plt.imread(str(cp))
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor('white')
                ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])
                ax.imshow(img)
                ax.axis('off')
                fig.text(0.07, 0.96, 'SVC Inspeção de Molas — DUAL', fontsize=16, fontweight='bold', va='top')
                fig.text(0.07, 0.93, 'Relatório de Auditoria — Gráfico', fontsize=12, va='top')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    st.session_state['last_report_pdf'] = str(pdf_path)
    st.session_state['last_report_html'] = str(html_path)
    st.session_state['last_report_generated_at'] = snapshot['generated_at']
    return pdf_path, html_path


def _parse_email_list(raw: str) -> list[str]:
    items = []
    for part in re.split(r'[;,]+', str(raw or '')):
        part = part.strip()
        if part:
            items.append(part)
    return items


def _build_email_subject() -> str:
    prefix = str(st.session_state.get("email_subject_prefix", "[SVC] Relatório de Auditoria")).strip()
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    line_name = str(st.session_state.get("line_name", "L01")).strip() or "L01"
    return f"{prefix} Relatório de Auditoria - {line_name} - {when}"


def send_report_email(pdf_path: str | Path, html_path: str | Path | None = None) -> tuple[bool, str]:
    pdf_path = Path(pdf_path)
    html_path = Path(html_path) if html_path else None

    if not pdf_path.exists():
        return False, f"PDF não encontrado: {pdf_path}"

    to_list = _parse_email_list(st.session_state.get("email_to", ""))
    cc_list = _parse_email_list(st.session_state.get("email_cc", ""))
    bcc_list = _parse_email_list(st.session_state.get("email_bcc", ""))
    recipients = to_list + cc_list + bcc_list
    if not recipients:
        return False, "Nenhum destinatário configurado em Para/CC/BCC."

    smtp_server = str(st.session_state.get("smtp_server", "")).strip()
    smtp_user = str(st.session_state.get("smtp_user", "")).strip()
    smtp_password = str(st.session_state.get("smtp_password", "")).strip()
    sender_name = str(st.session_state.get("email_sender_name", "SVC Inspeção de Molas")).strip() or "SVC Inspeção de Molas"
    use_tls = bool(st.session_state.get("smtp_use_tls", True))
    try:
        smtp_port = int(st.session_state.get("smtp_port", 587))
    except Exception:
        smtp_port = 587

    if not smtp_server:
        return False, "SMTP server não configurado."
    if not smtp_user:
        return False, "SMTP user não configurado."
    if not smtp_password:
        return False, "SMTP password não configurado."

    snapshot = report_summary_snapshot()
    subject = _build_email_subject()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{smtp_user}>"
    if to_list:
        msg["To"] = ", ".join(to_list)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)

    body = f"""Olá,

Segue em anexo o relatório de auditoria do SVC.

Emitido em: {snapshot['generated_at']}
Linha: {snapshot['line_name']}
Equipamento: {snapshot['equipment_id']}
Modelo: {snapshot['model_name']}
OP: {snapshot['production_order']}
Total: {snapshot['total']}
OK: {snapshot['ok']}
NG: {snapshot['ng']}
Yield: {snapshot['yield_pct']:.2f}%

Este e-mail foi gerado automaticamente pelo SVC.
"""
    msg.set_content(body)

    if html_path and html_path.exists():
        html_body = html_path.read_text(encoding='utf-8', errors='ignore')
        msg.add_alternative(html_body, subtype='html')

    for fp in [pdf_path, html_path] if html_path else [pdf_path]:
        if fp and Path(fp).exists():
            fp = Path(fp)
            ctype, _ = mimetypes.guess_type(str(fp))
            if not ctype:
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            with open(fp, 'rb') as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=fp.name)

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as server:
            server.ehlo()
            if use_tls:
                context = ssl.create_default_context()
                server.starttls(context=context)
                server.ehlo()
            server.login(smtp_user, smtp_password)
            server.send_message(msg, to_addrs=recipients)
        return True, f"Relatório enviado com sucesso para {len(recipients)} destinatário(s)."
    except Exception as e:
        return False, str(e)




# ==========================================================
# AUTOMAÇÃO DE RELATÓRIOS (JSON)
# ==========================================================
def load_auto_report_config() -> dict:
    try:
        if AUTO_REPORT_CONFIG_PATH.exists():
            data = json.loads(AUTO_REPORT_CONFIG_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}

def save_auto_report_history(payload: dict) -> None:
    AUTO_REPORT_HISTORY_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def load_auto_report_history() -> dict:
    try:
        if AUTO_REPORT_HISTORY_PATH.exists():
            data = json.loads(AUTO_REPORT_HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {"ultimo_envio": {"turno_1": None, "turno_2": None, "turno_3": None, "daily": None, "weekly": None, "monthly": None}, "ultima_atualizacao": None}

def _day_name_pt(dt: datetime) -> str:
    return ["segunda","terca","quarta","quinta","sexta","sabado","domingo"][dt.weekday()]

def _parse_hhmm(s: str) -> tuple[int,int] | None:
    try:
        hh, mm = str(s).strip().split(":", 1)
        hh = int(hh); mm = int(mm)
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return hh, mm
    except Exception:
        pass
    return None

def _now_auto_report() -> datetime:
    return datetime.now()

def _make_auto_key(kind: str, now: datetime) -> str:
    if kind.startswith("turno_") or kind == "daily":
        return now.strftime("%Y-%m-%d")
    if kind == "weekly":
        y, w, _ = now.isocalendar()
        return f"{y}-W{int(w):02d}"
    if kind == "monthly":
        return now.strftime("%Y-%m")
    return now.strftime("%Y-%m-%d %H:%M")

def _is_time_match(now: datetime, hhmm: str, window_min: int = 2) -> bool:
    parsed = _parse_hhmm(hhmm)
    if not parsed:
        return False
    hh, mm = parsed
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    diff = abs((now - target).total_seconds())
    return diff <= window_min * 60

def _auto_send_report(reason_label: str) -> tuple[bool, str]:
    try:
        pdf_path, html_path = generate_audit_report_files()
    except Exception as e:
        return False, f"Falha ao gerar relatório automático ({reason_label}): {e}"
    try:
        ok, msg = send_report_email(pdf_path, html_path)
        if ok:
            return True, f"Envio automático OK ({reason_label})"
        return False, f"Falha no envio automático ({reason_label}): {msg}"
    except Exception as e:
        return False, f"Falha no envio automático ({reason_label}): {e}"

def check_auto_report_schedule() -> None:
    cfg = load_auto_report_config()
    hist = load_auto_report_history()
    st.session_state["auto_reports_cfg_loaded"] = bool(cfg)
    st.session_state["auto_reports_enabled"] = bool(cfg.get("auto_send_enabled", False)) if cfg else False

    now = _now_auto_report()
    st.session_state["auto_reports_last_check"] = now.strftime("%d/%m/%y %H:%M:%S")

    if not cfg:
        st.session_state["auto_reports_status_msg"] = "JSON de automação não encontrado."
        return
    if not bool(cfg.get("auto_send_enabled", False)):
        st.session_state["auto_reports_status_msg"] = "Auto envio desativado no JSON."
        return
    if not bool(st.session_state.get("email_reports_enabled", False)):
        st.session_state["auto_reports_status_msg"] = "Auto envio ativo no JSON, mas e-mail está desabilitado no app."
        return

    ultimo = hist.setdefault("ultimo_envio", {})

    # turnos
    shift_cfg = cfg.get("shift_reports", {}) if isinstance(cfg.get("shift_reports", {}), dict) else {}
    if bool(shift_cfg.get("enabled", False)):
        for turno_key in ["turno_1", "turno_2", "turno_3"]:
            tcfg = shift_cfg.get(turno_key, {}) if isinstance(shift_cfg.get(turno_key, {}), dict) else {}
            if not bool(tcfg.get("ativo", False)):
                continue
            if not bool(tcfg.get("enviar_ao_final", True)):
                continue
            dias = tcfg.get("dias", []) or []
            if dias and _day_name_pt(now) not in dias:
                continue
            if not _is_time_match(now, str(tcfg.get("fim", ""))):
                continue
            key = _make_auto_key(turno_key, now)
            if str(ultimo.get(turno_key)) == key:
                continue
            ok, msg = _auto_send_report(turno_key)
            st.session_state["auto_reports_status_msg"] = msg
            if ok:
                ultimo[turno_key] = key
                hist["ultima_atualizacao"] = now.isoformat(timespec="seconds")
                save_auto_report_history(hist)
            return

    # diário
    daily_cfg = cfg.get("daily_report", {}) if isinstance(cfg.get("daily_report", {}), dict) else {}
    if bool(daily_cfg.get("enabled", False)) and _is_time_match(now, str(daily_cfg.get("horario_envio", ""))):
        key = _make_auto_key("daily", now)
        if str(ultimo.get("daily")) != key:
            ok, msg = _auto_send_report("daily")
            st.session_state["auto_reports_status_msg"] = msg
            if ok:
                ultimo["daily"] = key
                hist["ultima_atualizacao"] = now.isoformat(timespec="seconds")
                save_auto_report_history(hist)
            return

    weekly_cfg = cfg.get("weekly_report", {}) if isinstance(cfg.get("weekly_report", {}), dict) else {}
    if bool(weekly_cfg.get("enabled", False)) and str(weekly_cfg.get("dia_envio", "")).strip() == _day_name_pt(now) and _is_time_match(now, str(weekly_cfg.get("horario_envio", ""))):
        key = _make_auto_key("weekly", now)
        if str(ultimo.get("weekly")) != key:
            ok, msg = _auto_send_report("weekly")
            st.session_state["auto_reports_status_msg"] = msg
            if ok:
                ultimo["weekly"] = key
                hist["ultima_atualizacao"] = now.isoformat(timespec="seconds")
                save_auto_report_history(hist)
            return

    monthly_cfg = cfg.get("monthly_report", {}) if isinstance(cfg.get("monthly_report", {}), dict) else {}
    try:
        monthly_day = int(monthly_cfg.get("dia_envio", 0))
    except Exception:
        monthly_day = 0
    if bool(monthly_cfg.get("enabled", False)) and monthly_day == now.day and _is_time_match(now, str(monthly_cfg.get("horario_envio", ""))):
        key = _make_auto_key("monthly", now)
        if str(ultimo.get("monthly")) != key:
            ok, msg = _auto_send_report("monthly")
            st.session_state["auto_reports_status_msg"] = msg
            if ok:
                ultimo["monthly"] = key
                hist["ultima_atualizacao"] = now.isoformat(timespec="seconds")
                save_auto_report_history(hist)
            return

    if not st.session_state.get("auto_reports_status_msg"):
        st.session_state["auto_reports_status_msg"] = "Automação carregada. Aguardando próximo horário configurado."

# ==========================================================
# ==========================================================
# STREAMLIT APP
# ==========================================================
st.set_page_config(page_title="SVC Inspeção de Molas — DUAL", layout="wide")
st.markdown("""
<div class="app-title-fixed">
    SVC Inspeção de Molas — DUAL
</div>
""", unsafe_allow_html=True)
# ==========================================================
# SERIAL (Arduino) — controle manual + auto-refresh seguro
# ==========================================================
# Import opcional (recomendado) para permitir que o sensor dispare sem clique.
# pip install streamlit-autorefresh
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

with st.sidebar:
    # dica de diagnóstico rápido
    if str(st.session_state.get("serial_status","")).startswith("ERR:"):
        st.error("Falha ao abrir a porta serial. Feche o Serial Monitor/IDE, verifique a COM correta e tente novamente.")

# WATCHDOG: evita travamento eterno
if st.session_state.get("capture_busy", False):
    t0 = float(st.session_state.get("capture_busy_since", 0.0))

    if t0 <= 0:
        st.session_state["capture_busy_since"] = time.time()

    elif (time.time() - t0) > 5.0:
        # libera captura
        st.session_state["capture_busy"] = False
        st.session_state["capture_busy_since"] = 0.0

        # ✅ limpa estados do sensor também
        st.session_state["sensor_job_pending"] = False
        st.session_state["pending_trigger"] = False

else:
    st.session_state["capture_busy_since"] = 0.0

# Auto-refresh leve (somente se Serial ON e pacote disponível)
if ((st.session_state.get("serial_on", False) and st.session_state.get("serial_autorefresh", True)) or st.session_state.get("auto_reports_enabled", False)) and HAS_AUTOREFRESH:
    # Mantém o app "respirando" SEMPRE, inclusive se capture_busy travar.
    # Assim o watchdog e o poll do serial continuam rodando.
    if st.session_state.get("capture_busy", False):
        interval = 1800  # (industrial) evita corrida de mídia durante inferência
    else:
        interval = 2200 if st.session_state.get("sensor_job_pending", False) else 3000

# Auto-refresh leve (desativado para evitar reset de contadores)
# st_autorefresh(interval=interval, key="serial_autorefresh_key")

import time

if "last_auto_check" not in st.session_state:
    st.session_state.last_auto_check = 0

if time.time() - st.session_state.last_auto_check > 30:
    check_auto_report_schedule()
    st.session_state.last_auto_check = time.time()

poll_serial_events_and_maybe_trigger()
check_auto_report_schedule()

# ==========================================================
# PONTE: pending_trigger (Serial) -> sensor_job_pending (job)
# ==========================================================
ss = st.session_state
if ss.get("pending_trigger", False) and not ss.get("sensor_job_pending", False):
    # arma UM job de inspeção (sem travar UI)
    now = time.time()
    settle_ms = int(ss.get("sensor_settle_ms", 220))
    ss["sensor_job_pending"] = True
    ss["sensor_job_kind"] = "sensor"
    ss["sensor_job_armed_ts"] = now
    ss["sensor_job_ready_at"] = now + (settle_ms / 1000.0)
    ss["pending_trigger"] = False
    ss["pending_trigger_src"] = None

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
            
/* ================================
   ANDON ALERT STYLE
================================ */

@keyframes andonBlink {
    0% { background-color: #ff4b4b; }
    50% { background-color: #8b0000; }
    100% { background-color: #ff4b4b; }
}

.andon-banner {
    animation: andonBlink 1s infinite;
    color: white;
    padding: 20px;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    border-radius: 10px;
    margin-bottom: 20px;
}

/* ================================ */
            
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
.pie-wrap{
    margin-top: -120px;
    margin-bottom: -40px;
    display:flex;
    justify-content:center;
    align-items:flex-start;
}
section[data-testid="stSidebar"] { padding-top: 6px; padding-bottom: 6px; }
section[data-testid="stSidebar"] label { margin-bottom: 1px !important; font-size: 11px; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] select { min-height: 32px !important; padding-top: 4px !important; padding-bottom: 4px !important; font-size: 14px; }
section[data-testid="stSidebar"] button { min-height: 32px !important; padding-top: 2px !important; padding-bottom: 2px !important; font-size: 12px; }
section[data-testid="stSidebar"] hr { margin-top: 6px; margin-bottom: 6px; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { margin: 6px 0 4px 0 !important; font-size: 16px !important; }
.block-container {
    padding-top: 1.35rem !important;
    padding-bottom: 0.4rem !important;
}

h1 { margin-top: 0 !important; margin-bottom: 0 !important; }
div[data-testid="stMarkdownContainer"] { overflow: visible !important; }
div[data-testid="element-container"] { overflow: visible !important; }
div[data-testid="stVerticalBlock"] { overflow: visible !important; }           
.kpi-label { font-size:10px; }
.kpi-value { font-size:16px; }
.kpi-wide { min-height:30px; padding:5px 8px; }
.kpi-value-yield { font-size:16px; }
            
 .app-title-wrap{
    width: 100%;
    display: block;
    margin: 0 0 10px 0;
    padding: 4px 0 2px 0;
    overflow: visible !important;
}

.app-title{
    font-size: 28px;
    font-weight: 800;
    line-height: 1.20;
    color: #111827;
    margin: 0;
    padding: 0;
    white-space: nowrap;
    overflow: visible !important;
}

.block-container {
    padding-top: 2.20rem !important;
    padding-bottom: 0.4rem !important;
}

div[data-testid="stMarkdownContainer"] { overflow: visible !important; }
div[data-testid="element-container"] { overflow: visible !important; }
div[data-testid="stVerticalBlock"] { overflow: visible !important; }           

.app-title-fixed{
    display: block;
    width: 100%;
    font-size: 30px;
    font-weight: 800;
    line-height: 1.25;
    color: #111827;
    margin-top: 18px !important;
    margin-bottom: 12px !important;
    padding-top: 14px !important;
    padding-bottom: 4px !important;
    white-space: nowrap;
    overflow: visible !important;
    position: relative;
    top: 0;
}            

.result-box{
    border-radius: 10px;
    padding: 16px;
    margin-top: -48px !important;
    margin-bottom: 14px;
    text-align: center;
}

div[data-testid="stPyplot"]{
    margin-top:-40px !important;
}

</style>""", unsafe_allow_html=True)


render_production_dashboard()
# ==========================================================
# SIDEBAR — LOGO + SOBRE
# ==========================================================
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo_empresa.jpg"

with st.sidebar:
    if LOGO_PATH.exists():
        try:
            st.image(str(LOGO_PATH), width="stretch")
        except Exception:
            # evita quebra por MediaFileStorageError em reruns rápidos
            st.caption("⚠️ Falha ao renderizar logo (streamlit media cache).")
    else:
        st.caption("⚠️ Logo não encontrado em assets/logo_empresa.jpg")

    st.markdown("---")

    
    with st.expander("ℹ️ Sobre o Sistema", expanded=False):
        st.markdown("### Sistema: SVC – Computer Vision System for Spring Inspection")
        st.markdown("- **Versão:** v1.0.1")
        st.markdown("- **Status:** Industrial Ready Version")
        st.markdown("- **Release:** 28/03/2026")
        st.markdown("- **Empresa:** Salcomp")
        st.markdown("- **Engenheiro Responsável:** André Gama de Matos")
        st.markdown("- **Orientador Acadêmico:** Prof. Dr. Carlos Maurício Seródio Figueiredo")
        st.markdown("- **Co-Orientador Acadêmico:** Prof. Dr. Jozias Parente de Oliveira")
        st.markdown("- **Programa de Pós-Graduação:** Mestrado em Engenharia Elétrica")
        st.markdown("- **Ênfase:** Sistemas Embarcados e Visão Computacional")
        st.markdown("- **Instituição:** Universidade do Estado do Amazonas – UEA")
        st.markdown("- **Unidade:** Escola Superior de Tecnologia – EST")
        st.markdown("**Desenvolvido nos Laboratórios de Sistemas Embarcados e Visão Computacional da Escola Superior de Tecnologia – UEA**")
        st.markdown("---")
        st.markdown("**Ambiente de Execução do Sistema**")
        st.markdown(f"- **Sistema Operacional:** {platform.system()} {platform.release()}")
        st.markdown(f"- **Python:** {platform.python_version()}")
        st.markdown(f"- **OpenCV:** {cv2.__version__}")
        st.markdown(f"- **TensorFlow:** {tf.__version__}")


    with st.expander("🛠 Debug Serial", expanded=False):
            st.caption("Painel de diagnóstico da comunicação Serial e gatilho do sensor.")
            st.divider()
            st.header("Serial (Arduino)")

            ports = list_com_ports()
            st.session_state.serial_port = st.selectbox(
                "Porta",
                options=ports,
                index=ports.index(st.session_state.serial_port) if st.session_state.serial_port in ports else 0,
            )

            baud_options = [115200, 57600, 9600]
            current_baud = int(st.session_state.get("serial_baud", 115200))
            if current_baud not in baud_options:
                current_baud = 115200
            st.session_state.serial_baud = st.selectbox(
                "Baud",
                options=baud_options,
                index=baud_options.index(current_baud),
            )

            st.session_state["serial_autorefresh"] = st.checkbox(
                "Auto-refresh (sensor sem clique)",
                value=bool(st.session_state.get("serial_autorefresh", True)),
                help="Mantém o app 'respirando' para processar eventos da Serial sem clique.",
            )

            cols1, cols2 = st.columns(2)

            with cols1:
                if st.button("Serial ON", width="stretch"):
                    try:
                        serial_start()
                    except Exception:
                        st.error("Falha ao abrir a porta serial. Feche o Serial Monitor/IDE, verifique a COM correta e tente novamente.")

            with cols2:
                if st.button("Serial OFF", width="stretch"):
                    serial_stop()

            st.caption(f"Status: {'ON' if st.session_state.get('serial_on', False) else 'OFF'}")

            th = st.session_state.get("serial_thread")
            st.caption(f"thread alive: {bool(th) and th.is_alive()}")

            q = st.session_state.get("serial_q", None)
            try:
                st.caption(f"queue size: {q.qsize() if q else 'NA'}")
            except Exception:
                st.caption("queue size: ?")

            st.caption(f"Último PRESENT: {st.session_state.get('serial_last_present', None)}")
            st.caption(f"pending_trigger: {st.session_state.get('pending_trigger', False)} | job_pending: {st.session_state.get('sensor_job_pending', False)}")
            st.caption(f"last_sensor_fire: {st.session_state.get('last_sensor_fire_status', '')} | err: {st.session_state.get('last_sensor_fire_error', '')}")
            st.caption(f"Trigger mode: {st.session_state.get('serial_trigger_mode', '')}")
            now_dbg = time.time()
            st.caption(f"capture_busy: {st.session_state.get('capture_busy', False)}")
            st.caption(f"ready_at - now: {st.session_state.get('sensor_job_ready_at', 0.0) - now_dbg:.3f}s")

            st.divider()
            if st.button("TESTE: Disparar 1x (simula sensor)", width="stretch"):
                _now = time.time()
                st.session_state["sensor_job_pending"] = True
                st.session_state["sensor_job_kind"] = "sensor"
                st.session_state["sensor_job_armed_ts"] = _now
                st.session_state["sensor_job_ready_at"] = _now + (float(st.session_state.get("sensor_settle_ms", 220)) / 1000.0)
                st.session_state["last_sensor_fire_status"] = "arming...(manual test)"
                st.session_state["last_sensor_fire_error"] = ""

                st.session_state["serial_trigger_mode"] = st.selectbox(
                "Modo de disparo",
                options=["stable_high", "press_0to1", "release_1to0"],
                index=["stable_high", "press_0to1", "release_1to0"].index(
                    st.session_state.get("serial_trigger_mode", "press_0to1")
                ),
                help="stable_high dispara 1x quando PRESENT=1 fica estável por N ms e rearma quando volta a 0.",
            )

            if st.button("RESET SENSOR STATE ⚠",type="primary", width="stretch"):
                st.session_state["sensor_job_pending"] = False
                st.session_state["pending_trigger"] = False
                st.session_state["capture_busy"] = False
                st.session_state["capture_busy_since"] = 0.0
                st.rerun()


# ==========================================================
# SIDEBAR — MODO + PIN
# ==========================================================
st.sidebar.header("Modo")
c1, c2 = st.sidebar.columns(2)

with c1:
    if st.sidebar.button("👷 Operador", width='stretch'):
        st.session_state["user_mode"] = "OPERADOR"
        st.session_state["eng_unlocked"] = False
        st.rerun()

with c2:
    if st.sidebar.button("🛠 Eng.", width='stretch'):
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

st.sidebar.markdown("---")
st.sidebar.subheader("Status do Equipamento")
st.sidebar.caption(f"Serial: {st.session_state.get('serial_status', 'OFF')}")
st.sidebar.caption(f"Sensor present: {st.session_state.get('sensor_present', False)}")
st.sidebar.caption(f"Last serial: {st.session_state.get('serial_last_line', '') or '---'}")
# ==========================================================
# SIDEBAR — PRODUÇÃO (MODELO/LINHA)
# ==========================================================
ensure_registry_file()
registry = load_registry(REGISTRY_PATH)
ativos = get_active_models(registry)

st.sidebar.subheader("Produção")

st.session_state["mes_enabled"] = st.sidebar.checkbox(
    "Ativar MES",
    value=bool(st.session_state.get("mes_enabled", False)),
    help="Quando ativo, exige OP + Equipment ID + Serial/QRCode e gera XML para integração."
)

trace_ui_value = st.sidebar.checkbox(
    "Ativar rastreabilidade por Serial / QRCode",
    value=bool(st.session_state.get("traceability_enabled", False)),
    help="Permite registrar número de série mesmo com MES desligado."
)
if st.session_state.get("mes_enabled", False):
    st.session_state["traceability_enabled"] = True
else:
    st.session_state["traceability_enabled"] = bool(trace_ui_value)

st.session_state["production_order"] = st.sidebar.text_input(
    "Ordem de Produção",
    value=str(st.session_state.get("production_order", "")),
    placeholder="Ex.: BK4338BRI_Y25",
    help="Obrigatória quando MES estiver ativo."
)

st.session_state["equipment_id"] = st.sidebar.text_input(
    "Equipment ID",
    value=str(st.session_state.get("equipment_id", "SVC01")),
    placeholder="Ex.: SVC01",
    disabled=(not is_eng),
    help="Identificador único da estação. Edição liberada no modo Engenharia."
)

st.session_state["serial_qr_code"] = st.sidebar.text_input(
    "Número de Série / QRCode",
    value=str(st.session_state.get("serial_qr_code", "")),
    placeholder="Ex.: GH44-03133A-R37Y9RX1GN8SC3",
    disabled=(not bool(st.session_state.get("traceability_enabled", False))),
    help="Obrigatório quando rastreabilidade estiver ativa."
)

_mes_txt = "ATIVO" if st.session_state.get("mes_enabled", False) else "DESLIGADO"
_trace_txt = "ATIVA" if st.session_state.get("traceability_enabled", False) else "DESLIGADA"
st.sidebar.caption(f"MES: {_mes_txt}")
st.sidebar.caption(f"Rastreabilidade: {_trace_txt}")
st.sidebar.caption(f"OP atual: {st.session_state.get('production_order', '').strip() or '---'}")
st.sidebar.caption(f"Equipment ID: {st.session_state.get('equipment_id', '').strip() or '---'}")
st.sidebar.caption(f"Serial atual: {st.session_state.get('serial_qr_code', '').strip() or '---'}")

st.sidebar.checkbox("Usar MobileNetV2 (PRODUÇÃO)", value=bool(st.session_state.get("use_mnv2_prod", True)), key="use_mnv2_prod")
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

if is_eng:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📧 Configuração de E-mails")
    st.sidebar.checkbox(
        "Habilitar relatórios por e-mail",
        key="email_reports_enabled",
        value=bool(st.session_state.get("email_reports_enabled", False)),
        help="Ativa o cadastro e o uso futuro do envio automático/manual de relatórios por e-mail."
    )
    st.sidebar.checkbox(
        "Enviar também ao gerar relatório",
        key="email_send_on_generate",
        value=bool(st.session_state.get("email_send_on_generate", False)),
        help="Quando o envio real estiver habilitado, permite disparar e-mail logo após gerar o relatório."
    )
    st.sidebar.checkbox(
        "Habilitar envio diário automático",
        key="email_auto_daily_enabled",
        value=bool(st.session_state.get("email_auto_daily_enabled", False)),
        help="Define a intenção de envio diário automático no horário configurado."
    )
    st.sidebar.text_input(
        "Horário do envio diário",
        value=str(st.session_state.get("email_daily_time", "17:30")),
        key="email_daily_time",
        placeholder="17:30",
        help="Formato HH:MM. Ex.: 17:30"
    )
    st.sidebar.text_input(
        "Para",
        value=str(st.session_state.get("email_to", "")),
        key="email_to",
        placeholder="qualidade@empresa.com; gerente@empresa.com",
        help="Separe múltiplos e-mails por ponto e vírgula."
    )
    st.sidebar.text_input(
        "CC",
        value=str(st.session_state.get("email_cc", "")),
        key="email_cc",
        placeholder="engenharia@empresa.com",
        help="Opcional. Separe múltiplos e-mails por ponto e vírgula."
    )
    st.sidebar.text_input(
        "BCC",
        value=str(st.session_state.get("email_bcc", "")),
        key="email_bcc",
        placeholder="",
        help="Opcional. Separe múltiplos e-mails por ponto e vírgula."
    )
    st.sidebar.text_input(
        "Prefixo do assunto",
        value=str(st.session_state.get("email_subject_prefix", "[SVC] Relatório de Auditoria")),
        key="email_subject_prefix",
        help="Ex.: [SVC] Relatório de Auditoria"
    )
    st.sidebar.text_input(
        "Nome do remetente",
        value=str(st.session_state.get("email_sender_name", "SVC Inspeção de Molas")),
        key="email_sender_name",
        help="Nome exibido no e-mail. Ex.: SVC Inspeção de Molas"
    )
    st.sidebar.text_input(
        "SMTP server",
        value=str(st.session_state.get("smtp_server", "smtp.office365.com")),
        key="smtp_server",
        help="Ex.: smtp.office365.com"
    )
    st.sidebar.number_input(
        "SMTP port",
        min_value=1,
        max_value=65535,
        step=1,
        value=int(st.session_state.get("smtp_port", 587)),
        key="smtp_port"
    )
    st.sidebar.text_input(
        "SMTP user",
        value=str(st.session_state.get("smtp_user", "")),
        key="smtp_user",
        help="Conta de envio. A senha/token deve ficar fora do código, em arquivo local protegido ou variável de ambiente."
    )
    st.sidebar.text_input(
        "SMTP password",
        value=str(st.session_state.get("smtp_password", "")),
        key="smtp_password",
        type="password",
        help="Senha de app/token SMTP. Será salva apenas no arquivo local de configuração."
    )
    st.sidebar.checkbox(
        "Usar TLS",
        value=bool(st.session_state.get("smtp_use_tls", True)),
        key="smtp_use_tls"
    )

    _email_autosaved = persist_email_settings_if_needed(force=False)

    c_mail1, c_mail2 = st.sidebar.columns(2)
    with c_mail1:
        if st.button("💾 Salvar e-mail", key="btn_save_email_cfg", use_container_width=True):
            try:
                persist_email_settings_if_needed(force=True)
                st.sidebar.success("Configuração de e-mail salva ✅")
            except Exception as e:
                st.sidebar.error(f"Falha ao salvar config de e-mail: {e}")
    with c_mail2:
        if st.button("🔄 Recarregar", key="btn_reload_email_cfg", use_container_width=True):
            try:
                st.session_state["email_config_reload_pending"] = True
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Falha ao recarregar config de e-mail: {e}")

    if st.session_state.get("email_config_reload_notice"):
        st.sidebar.success(st.session_state.get("email_config_reload_notice"))
        st.session_state["email_config_reload_notice"] = ""
    if _email_autosaved:
        st.sidebar.caption("Dados de e-mail salvos automaticamente nos JSONs locais.")
    elif st.session_state.get("email_bootstrap_ok", False):
        st.sidebar.caption("Configuração carregada automaticamente dos JSONs locais.")
    if st.session_state.get("email_last_save_error"):
        st.sidebar.warning(f"Falha ao salvar e-mail automaticamente: {st.session_state.get('email_last_save_error')}")
        st.session_state["email_last_save_error"] = ""
    st.sidebar.caption(email_status_summary())
    hist_name = AUTO_REPORT_HISTORY_PATH.name if AUTO_REPORT_HISTORY_PATH.exists() else "historico_envio_relatorios.json"
    st.sidebar.caption(f"Histórico: {hist_name}")
    st.sidebar.caption(f"Auto envio ativo: {bool(st.session_state.get('auto_reports_enabled', False))}")
    if st.session_state.get("auto_reports_cfg_loaded", False):
        st.sidebar.caption("Configuração carregada automaticamente dos JSONs locais.")
    st.sidebar.caption(f"Arquivos locais: {EMAIL_CONFIG_PATH.name} | {EMAIL_CONTACTS_PATH.name}")

# ==========================================================
# PRELOAD DO MODELO (NÃO mexe em câmera; evita sensor inferir sem modelo)
# ==========================================================
if st.session_state.get("serial_on", False):
    try:
        # só tenta se ainda não está carregado
        if st.session_state.get("model") is None or st.session_state.get("labels") is None:
            ensure_active_model_loaded_or_raise(blocking=True)
    except Exception as e:
        # não trava UI — só registra
        st.session_state["last_error"] = f"Falha ao carregar modelo (pré-load): {e}"

st.sidebar.text_input(
    "Linha",
    value=str(st.session_state.get("line_name", "L01")),
    key="line_name"
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

st.sidebar.caption(f"Último Inspection ID: {st.session_state.get('last_inspection_id', '') or '---'}")
st.sidebar.caption(f"Status integração: {st.session_state.get('last_mes_status', 'LOCAL')}")
last_xml_name = Path(st.session_state.get('last_xml_path', '')).name if st.session_state.get('last_xml_path', '') else '---'
st.sidebar.caption(f"Último XML: {last_xml_name}")
if st.session_state.get("auto_reports_status_msg"):
    st.sidebar.caption(f"Auto-relatórios: {st.session_state.get('auto_reports_status_msg')}")

if st.sidebar.button("🔄 Reset Produção", use_container_width=True):
    st.session_state["cnt_total"] = 0
    st.session_state["cnt_ok"] = 0
    st.session_state["cnt_ng"] = 0
    st.session_state["cnt_ng_esq"] = 0
    st.session_state["cnt_ng_dir"] = 0
    st.session_state["history"] = []
    st.session_state["cnt_missing_esq"] = 0
    st.session_state["cnt_missing_dir"] = 0
    st.session_state["cnt_missing_both"] = 0
    st.session_state["cnt_misaligned_esq"] = 0
    st.session_state["cnt_misaligned_dir"] = 0
    st.session_state["cnt_misaligned_both"] = 0
    st.session_state["cnt_misto"] = 0
    st.session_state["cnt_ok_attention"] = 0
    st.session_state["production_started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["last_warning"] = "Contadores de produção resetados."
    st.rerun()

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

st.session_state["use_dshow"] = bool(use_dshow)

col_cam_btns = st.sidebar.columns(2)
with col_cam_btns[0]:
    btn_cam_on = st.sidebar.button("📷 Ligar", width='stretch')
with col_cam_btns[1]:
    btn_cam_off = st.sidebar.button("⛔ Desligar", width='stretch')

btn_capture = st.sidebar.button("📸 Capturar + Inferir (DUAL)", type="primary", width='stretch')
btn_live = st.sidebar.button("▶️ LIVE", width='stretch')

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
        btn_save_ok = st.sidebar.button("✅ Salvar OK", width='stretch')
    with c_ng:
        btn_save_ng = st.sidebar.button("❌ Salvar NG", width='stretch')

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

        btn_make_split = st.button("🚀 Gerar Split agora", width='stretch')

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
# SIDEBAR — SIMULAÇÃO POR UPLOAD (ENG)
# ==========================================================
if is_eng:
    st.sidebar.divider()
    st.sidebar.header("🖼️ Simulação por Upload")
    st.sidebar.caption("Use fotos do dataset ou imagens já capturadas para testar o SVC sem câmera/sensor.")

    uploaded_test_file = st.sidebar.file_uploader(
        "Enviar foto do produto",
        type=["jpg", "jpeg", "png", "bmp"],
        key="eng_upload_test_file",
        help="A imagem enviada será usada como entrada offline para a mesma lógica DUAL do SVC."
    )

    st.session_state["upload_test_count_kpi"] = st.sidebar.checkbox(
        "Contabilizar upload nos KPIs",
        value=bool(st.session_state.get("upload_test_count_kpi", False)),
        help="Deixe desligado para testes de laboratório sem impactar os contadores de produção."
    )

    if uploaded_test_file is not None:
        try:
            preview_bgr = decode_uploaded_image_to_bgr(uploaded_test_file)
            st.session_state["upload_test_frame"] = preview_bgr.copy()
            st.session_state["upload_test_name"] = str(uploaded_test_file.name)
            st.sidebar.caption(f"Arquivo atual: {uploaded_test_file.name}")
            st.sidebar.image(cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB), width=240, caption="Preview do upload")
        except Exception as e:
            st.sidebar.error(f"Falha ao abrir imagem enviada: {e}")

    c_up1, c_up2 = st.sidebar.columns(2)
    with c_up1:
        if st.sidebar.button("🧪 Inspecionar upload", width='stretch'):
            if uploaded_test_file is None:
                st.sidebar.warning("Envie uma imagem primeiro.")
            else:
                try:
                    src_upload = decode_uploaded_image_to_bgr(uploaded_test_file)
                    run_infer_dual_on_uploaded_frame(
                        src_upload,
                        file_name=str(uploaded_test_file.name),
                        update_metrics=bool(st.session_state.get("upload_test_count_kpi", False)),
                    )
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Falha na inspeção por upload: {e}")
    with c_up2:
        if st.sidebar.button("🧹 Limpar upload", width='stretch'):
            st.session_state["upload_test_frame"] = None
            st.session_state["upload_test_name"] = ""
            st.rerun()

# ==========================================================
# SIDEBAR — EVIDÊNCIAS / RETENÇÃO (ENG)
# ==========================================================
if is_eng:
    st.sidebar.divider()
    st.sidebar.header("📸 Evidências / Auditoria")

    st.session_state["evidence_auto_enabled"] = st.sidebar.checkbox(
        "Ativar SALVAR ERROS AUTOMATICAMENTE",
        value=bool(st.session_state.get("evidence_auto_enabled", True)),
        help="Salva automaticamente NG_DESALINHADO, NG_FALTANDO e, opcionalmente, OK próximo do limite."
    )

    st.session_state["evidence_save_ok_limit"] = st.sidebar.checkbox(
        "Salvar também OK próximo do limite",
        value=bool(st.session_state.get("evidence_save_ok_limit", True)),
        help="Guarda evidências quando a peça foi aprovada, mas ficou na banda de atenção."
    )

    st.session_state["evidence_retention_enabled"] = st.sidebar.checkbox(
        "Ativar auto delete por retenção",
        value=bool(st.session_state.get("evidence_retention_enabled", True)),
        help="Remove arquivos antigos automaticamente para evitar crescimento excessivo em disco."
    )

    st.session_state["evidence_retention_days"] = st.sidebar.selectbox(
        "Auto delete após",
        options=[30, 60, 90],
        index=[30, 60, 90].index(int(st.session_state.get("evidence_retention_days", 60))),
    )

    st.session_state["evidence_warning_gb"] = st.sidebar.number_input(
        "Avisar quando pasta automática exceder (GB)",
        min_value=0.5, max_value=100.0,
        value=float(st.session_state.get("evidence_warning_gb", 5.0)),
        step=0.5,
    )

    _deleted_files, _deleted_bytes = maybe_cleanup_auto_evidence(
        AUTO_EVIDENCE_DIR,
        retention_days=int(st.session_state.get("evidence_retention_days", 60)),
        enabled=bool(st.session_state.get("evidence_retention_enabled", True)),
        interval_sec=1800,
    )

    auto_folder_bytes = folder_size_bytes(AUTO_EVIDENCE_DIR)
    auto_folder_files = count_evidence_files(AUTO_EVIDENCE_DIR)
    disk_info = get_disk_status(BASE_DIR)
    disk_status, disk_label = disk_free_status_label(disk_info.get("free_gb", 0.0), warn_gb=10.0, critical_gb=5.0)

    st.sidebar.caption(f"Pasta automática: `{AUTO_EVIDENCE_DIR}`")
    st.sidebar.caption(f"Tamanho atual: **{bytes_to_human(auto_folder_bytes)}**")
    st.sidebar.caption(f"Arquivos atuais: **{auto_folder_files}**")
    st.sidebar.caption(f"Espaço livre em disco: **{disk_info.get('free_gb', 0.0):.2f} GB**")

    if auto_folder_bytes >= float(st.session_state.get("evidence_warning_gb", 5.0)) * 1024 * 1024 * 1024:
        st.sidebar.warning("Pasta automática acima do limite configurado.")
    else:
        st.sidebar.success("Pasta automática dentro do limite configurado.")

    if disk_status == "critical":
        st.sidebar.error(f"Disco em estado {disk_label}. Libere espaço para evitar travamentos.")
    elif disk_status == "warning":
        st.sidebar.warning(f"Disco em estado {disk_label}. Recomenda-se acompanhar o armazenamento.")
    else:
        st.sidebar.success(f"Disco em estado {disk_label}.")

    if bool(st.session_state.get("evidence_retention_enabled", True)):
        st.sidebar.info(f"Retenção ativa: {int(st.session_state.get('evidence_retention_days', 60))} dias.")
    else:
        st.sidebar.info("Retenção automática desativada.")

    if _deleted_files > 0:
        st.sidebar.success(f"Limpeza automática: {_deleted_files} arquivo(s) removido(s), liberando {bytes_to_human(_deleted_bytes)}.")

    if st.sidebar.button("🧹 Executar limpeza agora", use_container_width=True):
        df, db = cleanup_old_evidence(AUTO_EVIDENCE_DIR, int(st.session_state.get("evidence_retention_days", 60)))
        st.session_state["evidence_last_cleanup_files"] = df
        st.session_state["evidence_last_cleanup_bytes"] = db
        st.sidebar.success(f"Limpeza manual concluída: {df} arquivo(s), {bytes_to_human(db)} liberados.")

    _recent_auto = list_recent_files(AUTO_EVIDENCE_DIR, limit=5)
    if _recent_auto:
        with st.sidebar.expander("🕘 Últimos arquivos automáticos", expanded=False):
            for _p in _recent_auto:
                st.caption(_p.name)

# ==========================================================
# SIDEBAR — CONFIG (apenas ENG liberado)
# ==========================================================

show_debug = False
if is_eng:
    st.sidebar.divider()
    st.sidebar.header("Config (Eng.)")
# ==========================================================
# ENGENHARIA — AJUSTES DE CALIBRAÇÃO
# Sliders usados somente para setup e validação industrial.
#
# Ordem recomendada de ajuste:
# 1) threshold_presente  -> separar presente x faltando
# 2) threshold_ng_ok     -> definir faixa de OK seguro
# 3) threshold_ng_ng     -> definir faixa de NG seguro
# 4) threshold_margem    -> reduzir falso NG perto da fronteira
# ==========================================================

    st.slider("Threshold mínimo p/ MOLA PRESENTE", 0.0, 1.0,
              value=float(st.session_state.get("threshold_presente", DEFAULT_THRESH_PRESENTE)),
              step=0.01, key="threshold_presente")

    st.slider("Faixa NG — limite OK seguro", 0.0, 1.0,
              value=float(st.session_state.get("threshold_ng_ok", DEFAULT_THR_NG_OK)),
              step=0.01, key="threshold_ng_ok")

    st.slider("Faixa NG — limite NG forte", 0.0, 1.0,
              value=float(st.session_state.get("threshold_ng_ng", DEFAULT_THR_NG_NG)),
              step=0.01, key="threshold_ng_ng")

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

    # tenta com backend escolhido
    cap = cv2.VideoCapture(int(cam_index), backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # fallback: tenta sem backend explícito (alguns drivers falham no DSHOW)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        cap = cv2.VideoCapture(int(cam_index))
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
# ==========================================================
# Infer DUAL — Lógica Industrial (OK / NG_MISSING / NG_MISALIGNED)
# ==========================================================
def decide_misaligned_status(prob_ng: float, prob_ok: float, thr_ng_ok: float, thr_ng_ng: float, margin_abs: float = 0.10):
    """Decisão robusta para desalinhamento em 3 bandas sem quebrar o fluxo industrial.

    Retorna:
      defect_code: OK | NG_MISALIGNED
      decision_band: OK_SAFE | ATTENTION | NG_STRONG
      margin_ok_minus_ng: prob_ok - prob_ng
      attention_flag: bool
    """
    prob_ng = float(prob_ng)
    prob_ok = float(prob_ok)
    thr_ng_ok = float(thr_ng_ok)
    thr_ng_ng = float(thr_ng_ng)
    margin = float(prob_ok - prob_ng)

    strong_ok = (prob_ng <= thr_ng_ok)
    strong_ng = (prob_ng >= thr_ng_ng) and (margin <= -float(margin_abs))

    if strong_ng:
        return "NG_MISALIGNED", "NG_STRONG", margin, False
    if strong_ok:
        return "OK", "OK_SAFE", margin, False
    return "OK", "ATTENTION", margin, True

# ==========================================================
# INFERÊNCIA DUAL — LÓGICA INDUSTRIAL
#
# Esta função é o coração da decisão do sistema.
# Ela combina:
# 1) modelo legacy de presença/ausência de mola
# 2) modelo MobileNetV2 de desalinhamento
#
# Prioridade industrial da decisão:
#   NG_MISSING > NG_MISALIGNED > OK
#
# Regras de negócio sugeridas:
# - Se p_presente < threshold_presente:
#       NG_MISSING
# - Se p_ng <= threshold_ng_ok:
#       OK seguro
# - Se p_ng >= threshold_ng_ng:
#       NG_MISALIGNED
# - Se ficar entre os dois limiares:
#       zona de atenção / revisar / baixa confiança
#
# Se houver falso NG em campo, revisar primeiro:
# - thresholds
# - margem de decisão
# - ROI
# - foco/iluminação/distância da câmera
# ==========================================================
def infer_dual_on_frame(frame_bgr: np.ndarray):
    """Inferência DUAL com lógica industrial calibrada para chão de fábrica.

    Saídas industriais:
      - OK
      - NG_MISSING     (falta de mola)
      - NG_MISALIGNED  (mola desalinhada)

    Estratégia:
      1) Detecta MISSING via modelo legacy de presença (mola_presente/mola_ausente) usando threshold_presente.
      2) Detecta MISALIGNED via MobileNetV2 Produção por ROI.
      3) Usa duas bandas para NG (OK seguro / atenção / NG forte), evitando falso NG na borda.
      4) Prioridade industrial: NG_MISSING > NG_MISALIGNED > OK
    """

    # ----------------------
    # ROIs (percentuais)
    # ----------------------
    esq_x0 = int(st.session_state.get("roi_esq_x0", DEFAULT_ROI["ESQ"]["x0"]))
    esq_x1 = int(st.session_state.get("roi_esq_x1", DEFAULT_ROI["ESQ"]["x1"]))
    esq_y0 = int(st.session_state.get("roi_esq_y0", DEFAULT_ROI["ESQ"]["y0"]))
    esq_y1 = int(st.session_state.get("roi_esq_y1", DEFAULT_ROI["ESQ"]["y1"]))

    dir_x0 = int(st.session_state.get("roi_dir_x0", DEFAULT_ROI["DIR"]["x0"]))
    dir_x1 = int(st.session_state.get("roi_dir_x1", DEFAULT_ROI["DIR"]["x1"]))
    dir_y0 = int(st.session_state.get("roi_dir_y0", DEFAULT_ROI["DIR"]["y0"]))
    dir_y1 = int(st.session_state.get("roi_dir_y1", DEFAULT_ROI["DIR"]["y1"]))

    normalize_roi = bool(st.session_state.get("normalize_lab_equalize", DEFAULT_NORMALIZE_LAB))

    roi_esq = crop_roi_percent(frame_bgr, esq_x0, esq_x1, esq_y0, esq_y1)
    roi_dir = crop_roi_percent(frame_bgr, dir_x0, dir_x1, dir_y0, dir_y1)

    if normalize_roi:
        roi_esq = equalize_lab_bgr(roi_esq)
        roi_dir = equalize_lab_bgr(roi_dir)

    # ----------------------
    # 1) PRESENÇA (MISSING) — Legacy (mola_presente/mola_ausente)
    # ----------------------
    th_presente = float(st.session_state.get("threshold_presente", DEFAULT_THRESH_PRESENTE))

    p_pres_esq = 1.0
    p_pres_dir = 1.0
    cls_pres_esq = "mola_presente"
    cls_pres_dir = "mola_presente"
    probs_pres_esq = None
    probs_pres_dir = None

    try:
        ensure_model_loaded_or_raise(blocking=True)
        if st.session_state.get("model") is not None and st.session_state.get("labels") is not None:
            cls_pres_esq, conf_esq_, probs_pres_esq = predict_one(
                st.session_state["model"], st.session_state["labels"], roi_esq
            )
            cls_pres_dir, conf_dir_, probs_pres_dir = predict_one(
                st.session_state["model"], st.session_state["labels"], roi_dir
            )
            p_pres_esq = prob_of_class(st.session_state["labels"], probs_pres_esq, "mola_presente")
            p_pres_dir = prob_of_class(st.session_state["labels"], probs_pres_dir, "mola_presente")
    except Exception:
        pass

    missing_esq = (p_pres_esq < th_presente)
    missing_dir = (p_pres_dir < th_presente)

    # ----------------------
    # 2) MISALIGN — MobileNetV2 Produção (OK vs NG_MISALIGNED)
    # ----------------------
    use_mnv2 = bool(st.session_state.get("use_mnv2_prod", True))

    cls_mis_esq = "OK"
    cls_mis_dir = "OK"
    prob_ng_esq = 0.0
    prob_ng_dir = 0.0
    prob_ok_esq = 1.0
    prob_ok_dir = 1.0
    probs_mis_esq = None
    probs_mis_dir = None

    thr_ng_ok = float(st.session_state.get("threshold_ng_ok", DEFAULT_THR_NG_OK))
    thr_ng_ng = float(st.session_state.get("threshold_ng_ng", DEFAULT_THR_NG_NG))
    margin_abs = float(st.session_state.get("prod_margin_abs", 0.10))

    decision_band_esq = "MISSING" if missing_esq else "OK_SAFE"
    decision_band_dir = "MISSING" if missing_dir else "OK_SAFE"
    margin_esq = 1.0
    margin_dir = 1.0
    attention_esq = False
    attention_dir = False

    if use_mnv2:
        ensure_prod_model_loaded_or_raise(blocking=True)
        model = st.session_state.get("prod_model")
        class_names = st.session_state.get("prod_class_names")
        pos_idx = int(st.session_state.get("prod_pos_idx", 0))
        img_size = tuple(st.session_state.get("prod_img_size", (224, 224)))

        if model is None or class_names is None:
            raise RuntimeError("Modelo PRODUÇÃO não carregado.")

        if not missing_esq:
            _, prob_ng_esq, probs_mis_esq = infer_mobilenetv2_prod(
                roi_esq, model, class_names, pos_idx, thr_ng_ng, img_size=img_size
            )
            prob_ok_esq = float((probs_mis_esq or {}).get("OK", 1.0 - prob_ng_esq))
            cls_mis_esq, decision_band_esq, margin_esq, attention_esq = decide_misaligned_status(
                prob_ng_esq, prob_ok_esq, thr_ng_ok, thr_ng_ng, margin_abs=margin_abs
            )
        else:
            cls_mis_esq, prob_ng_esq, probs_mis_esq = ("OK", 0.0, None)

        if not missing_dir:
            _, prob_ng_dir, probs_mis_dir = infer_mobilenetv2_prod(
                roi_dir, model, class_names, pos_idx, thr_ng_ng, img_size=img_size
            )
            prob_ok_dir = float((probs_mis_dir or {}).get("OK", 1.0 - prob_ng_dir))
            cls_mis_dir, decision_band_dir, margin_dir, attention_dir = decide_misaligned_status(
                prob_ng_dir, prob_ok_dir, thr_ng_ok, thr_ng_ng, margin_abs=margin_abs
            )
        else:
            cls_mis_dir, prob_ng_dir, probs_mis_dir = ("OK", 0.0, None)

    mis_esq = (cls_mis_esq == "NG_MISALIGNED") and (not missing_esq)
    mis_dir = (cls_mis_dir == "NG_MISALIGNED") and (not missing_dir)

    defect_esq = "OK"
    defect_dir = "OK"

    if missing_esq:
        defect_esq = "NG_MISSING"
    elif mis_esq:
        defect_esq = "NG_MISALIGNED"

    if missing_dir:
        defect_dir = "NG_MISSING"
    elif mis_dir:
        defect_dir = "NG_MISALIGNED"

    if defect_esq == "NG_MISSING" or defect_dir == "NG_MISSING":
        defect_type = "NG_MISSING"
    elif defect_esq == "NG_MISALIGNED" or defect_dir == "NG_MISALIGNED":
        defect_type = "NG_MISALIGNED"
    else:
        defect_type = "OK"

    attention_flag = bool(attention_esq or attention_dir)
    aprovado = (defect_type == "OK")
    ok_esq = (defect_esq == "OK")
    ok_dir = (defect_dir == "OK")

    return {
        "roi_esq": roi_esq,
        "roi_dir": roi_dir,
        "cls_esq": cls_pres_esq,
        "cls_dir": cls_pres_dir,
        "p_pres_esq": float(p_pres_esq),
        "p_pres_dir": float(p_pres_dir),
        "conf_esq": float(p_pres_esq),
        "conf_dir": float(p_pres_dir),
        "cls_mnv2_esq": cls_mis_esq,
        "cls_mnv2_dir": cls_mis_dir,
        "prob_ng_esq": float(prob_ng_esq),
        "prob_ng_dir": float(prob_ng_dir),
        "prob_ok_esq": float(prob_ok_esq),
        "prob_ok_dir": float(prob_ok_dir),
        "thr_ng": float(thr_ng_ng),
        "thr_ng_ok": float(thr_ng_ok),
        "thr_ng_ng": float(thr_ng_ng),
        "margin_esq": float(margin_esq),
        "margin_dir": float(margin_dir),
        "decision_band_esq": decision_band_esq,
        "decision_band_dir": decision_band_dir,
        "attention_esq": bool(attention_esq),
        "attention_dir": bool(attention_dir),
        "attention_flag": attention_flag,
        "probs_esq": probs_mis_esq,
        "probs_dir": probs_mis_dir,
        "defect_esq": defect_esq,
        "defect_dir": defect_dir,
        "defect_type": defect_type,
        "ok_esq": ok_esq,
        "ok_dir": ok_dir,
        "aprovado": aprovado,
    }

def update_metrics_and_history(res: dict) -> None:
    """Atualiza contadores de produção e histórico (para gráficos)."""
    st.session_state["cnt_total"] = int(st.session_state.get("cnt_total", 0)) + 1

    aprovado = bool(res.get("aprovado", False))
    if aprovado:
        st.session_state["cnt_ok"] = int(st.session_state.get("cnt_ok", 0)) + 1
        if bool(res.get("attention_flag", False)):
            st.session_state["cnt_ok_attention"] = int(st.session_state.get("cnt_ok_attention", 0)) + 1
    else:
        st.session_state["cnt_ng"] = int(st.session_state.get("cnt_ng", 0)) + 1
        if not bool(res.get("ok_esq", True)):
            st.session_state["cnt_ng_esq"] = int(st.session_state.get("cnt_ng_esq", 0)) + 1
        if not bool(res.get("ok_dir", True)):
            st.session_state["cnt_ng_dir"] = int(st.session_state.get("cnt_ng_dir", 0)) + 1

        defect_esq = str(res.get("defect_esq", "OK") or "OK").strip().upper()
        defect_dir = str(res.get("defect_dir", "OK") or "OK").strip().upper()

        if defect_esq == "NG_MISSING" and defect_dir == "OK":
            st.session_state["cnt_missing_esq"] = int(st.session_state.get("cnt_missing_esq", 0)) + 1
        elif defect_esq == "OK" and defect_dir == "NG_MISSING":
            st.session_state["cnt_missing_dir"] = int(st.session_state.get("cnt_missing_dir", 0)) + 1
        elif defect_esq == "NG_MISSING" and defect_dir == "NG_MISSING":
            st.session_state["cnt_missing_both"] = int(st.session_state.get("cnt_missing_both", 0)) + 1
        elif defect_esq == "NG_MISALIGNED" and defect_dir == "OK":
            st.session_state["cnt_misaligned_esq"] = int(st.session_state.get("cnt_misaligned_esq", 0)) + 1
        elif defect_esq == "OK" and defect_dir == "NG_MISALIGNED":
            st.session_state["cnt_misaligned_dir"] = int(st.session_state.get("cnt_misaligned_dir", 0)) + 1
        elif defect_esq == "NG_MISALIGNED" and defect_dir == "NG_MISALIGNED":
            st.session_state["cnt_misaligned_both"] = int(st.session_state.get("cnt_misaligned_both", 0)) + 1
        else:
            st.session_state["cnt_misto"] = int(st.session_state.get("cnt_misto", 0)) + 1

    hist = st.session_state.get("history", [])
    hist.append({
        "n": int(st.session_state.get("cnt_total", 0)),
        "aprovado": int(aprovado),
        "ok_esq": int(bool(res.get("ok_esq", False))),
        "ok_dir": int(bool(res.get("ok_dir", False))),
        "p_esq": float(res.get("p_pres_esq", 0.0)),
        "p_dir": float(res.get("p_pres_dir", 0.0)),
        "ng_esq": int(not bool(res.get("ok_esq", True))),
        "ng_dir": int(not bool(res.get("ok_dir", True))),
        "defect_detail": build_defect_detail_code(res),
    })
    st.session_state["history"] = hist

# ==========================================================
# EXECUÇÃO DO TRIGGER DO SENSOR (FLUXO ÚNICO E BLINDADO)
# ==========================================================

def execute_sensor_job_if_ready():
    ss = st.session_state

    if not ss.get("sensor_job_pending", False):
        return

    if ss.get("capture_busy", False):
        return

    now = time.time()
    ready_at = float(ss.get("sensor_job_ready_at", 0.0))

    if now < ready_at:
        return

    # 🔒 trava captura
    ss["capture_busy"] = True
    ss["capture_busy_since"] = now

    try:
        ensure_active_model_loaded_or_raise(blocking=True)

        run_capture_infer_dual(trigger_source="sensor")

        ss["last_error"] = None
        ss["last_sensor_fire_status"] = "OK (infer done)"
        ss["last_sensor_fire_error"] = ""

    except Exception as e:
        ss["last_sensor_fire_status"] = "ERR"
        ss["last_sensor_fire_error"] = str(e)
        ss["last_error"] = f"Erro na inferência: {e}"

    finally:
        # 🔓 libera estados SEMPRE
        ss["sensor_job_pending"] = False
        ss["sensor_job_kind"] = None
        ss["capture_busy"] = False
        ss["capture_busy_since"] = 0.0


# ==========================================================
# ACTIONS — Capturar + Inferir
# ==========================================================
def run_capture_infer_dual(trigger_source: str = "button"):
    """Captura frame fresco (ou last_frame) e executa inferência DUAL.
    trigger_source: 'button' ou 'sensor' (vai para o log).
    """
    st.session_state["last_error"] = None
    st.session_state["last_warning"] = None

    # normaliza serial lido/digitado para manter consistência no log e no XML
    st.session_state["serial_qr_code"] = normalize_serial_qr(st.session_state.get("serial_qr_code", ""))

    ok_ctx, msg_ctx = validate_operation_context()
    if not ok_ctx:
        st.session_state["last_error"] = f"Inspeção bloqueada: {msg_ctx}"
        st.session_state["last_result"] = None
        return

    serial_number_ctx = normalize_serial_qr(st.session_state.get("serial_qr_code", ""))
    traceability_enabled_ctx = bool(st.session_state.get("traceability_enabled", False))
    mes_enabled_ctx = bool(st.session_state.get("mes_enabled", False))
    if traceability_enabled_ctx and serial_number_ctx:
        if check_serial_duplicate(serial_number_ctx, TRACE_LOG_PATH):
            if mes_enabled_ctx:
                st.session_state["last_error"] = f"Serial já inspecionado: {serial_number_ctx}. Bloqueado porque o MES está ativo."
                st.session_state["last_result"] = None
                return
            else:
                st.session_state["last_warning"] = f"Serial já inspecionado anteriormente: {serial_number_ctx}. Inspeção liberada porque o MES está desligado."

    # carrega modelo/labels SOMENTE aqui (evita tela branca na troca de modelo)
    try:
        # ✅ SEMPRE garante modelo/labels aqui (botão e sensor passam pelo mesmo funil)
        # blocking=True evita corrida com o lock quando o sensor dispara em reruns rápidos
        ensure_active_model_loaded_or_raise(blocking=True)
        # garantia extra (modelo certo conforme modo)
        if st.session_state.get("use_mnv2_prod", True):
            if st.session_state.get("prod_model") is None or st.session_state.get("prod_class_names") is None:
                raise RuntimeError("Modelo PRODUÇÃO não carregado (após ensure).")
        else:
            if st.session_state.get("model") is None or st.session_state.get("labels") is None:
                raise RuntimeError("Modelo LEGACY não carregado (após ensure).")
    except Exception as e:
        st.session_state["last_error"] = f"Falha ao carregar modelo atual: {e}"
        st.session_state["last_result"] = None
        return
    # pega frame
    src = None
    if st.session_state.get("camera_on") and st.session_state.get("cap") is not None:
        cap = st.session_state["cap"]
        try:
            if trigger_source == "sensor":
                # Sensor: 1 frame rápido com timeout (não trava UI)
                src = read_one_frame_timeout(cap, timeout_s=1.5)
            else:
                # Botão: flush para pegar frame mais fresco
                src = read_fresh_frame(
                    cap,
                    flush_grabs=12,
                    sleep_ms=10,
                    extra_reads=2
                )
        except Exception as e:
            st.session_state["last_error"] = f"Falha ao ler frame da câmera: {e}"
            src = None

        if src is not None:
            st.session_state["last_frame"] = src.copy()
    else:
        lf = st.session_state.get("last_frame")
        src = lf.copy() if lf is not None else None

    if src is None:
        st.session_state["last_error"] = "Sem imagem para inferir (ligue a câmera e capture)."
        st.session_state["last_result"] = None
        return

    # Frame usado na inferência
    st.session_state["display_frame"] = src.copy()
    # assinatura do frame inferido (para detectar troca de peça no modo automático)
    st.session_state["last_infer_sig"] = quick_frame_signature(src)
    st.session_state["last_infer_ts"] = time.time()

    st.session_state["last_frame"] = src.copy()

    # Tempo de teste (para log)
    start_dt = datetime.now()
    try:
        if trigger_source == "sensor":
            res = infer_dual_on_frame_timeout(src, timeout_s=4.0)
        else:
            print("[DEBUG] entrando na inferência...")
            res = infer_dual_on_frame(src)
            print("[DEBUG] inferência retornou.")
        end_dt = datetime.now()
        test_time_sec = (end_dt - start_dt).total_seconds()

        st.session_state["last_result"] = res
        update_metrics_and_history(res)
        auto_save_current_result_if_needed(res, frame_bgr=src, source=trigger_source)

        # 🔔 BIP quando NG
        if not res.get("aprovado", False):
            beep_ng()

        timestamp = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        th_local = float(st.session_state.get("threshold_presente", DEFAULT_THRESH_PRESENTE))
        cs_code, cs_detail = get_cs_code(res, th_local)

        total = int(st.session_state.get("cnt_total", 0))
        ok = int(st.session_state.get("cnt_ok", 0))
        ng = int(st.session_state.get("cnt_ng", 0))
        yield_pct = round((ok / total * 100.0), 2) if total > 0 else 0.0

        cam_index_local = int(st.session_state.get("cam_index_last", st.session_state.get("camera_index", 0)))
        use_dshow_local = bool(st.session_state.get("use_dshow", True))

        final_result = str(res.get("defect_type", "OK" if res.get("aprovado", False) else "NG"))
        result_left = str(res.get("defect_esq", "OK"))
        result_right = str(res.get("defect_dir", "OK"))
        conf_left = float(res.get("p_pres_esq", 0.0))
        conf_right = float(res.get("p_pres_dir", 0.0))
        image_stub = f"in_memory_cam{cam_index_local}_{end_dt.strftime('%Y%m%d_%H%M%S')}"

        row = {
            "timestamp": timestamp,
            "modelo": st.session_state.get("product_model", ""),
            "linha": st.session_state.get("line_name", ""),
            "resultado_final": final_result,
            "defect_esq": result_left,
            "defect_dir": result_right,
            "cs_code": cs_code,
            "cs_detail": cs_detail,
            "p_esq": round(conf_left, 4),
            "p_dir": round(conf_right, 4),
            "th_presente": float(th_local),
            "camera_index": cam_index_local,
            "directshow": use_dshow_local,
            "source": trigger_source,  # sensor/button
            "total": total,
            "ok": ok,
            "ng": ng,
            "yield_pct": yield_pct,
            "test_time_sec": f"{test_time_sec:.3f}",
            "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        }

        append_log_csv(row)

        inspection_id = generate_inspection_id()
        st.session_state["last_inspection_id"] = inspection_id

        system_name = str(st.session_state.get("system_name", "SVC Inspeção de Molas - DUAL"))
        equipment_id = str(st.session_state.get("equipment_id", "SVC01")).strip()
        mes_enabled = bool(st.session_state.get("mes_enabled", False))
        traceability_enabled = bool(st.session_state.get("traceability_enabled", False))
        production_order = str(st.session_state.get("production_order", "")).strip()
        serial_number = normalize_serial_qr(st.session_state.get("serial_qr_code", ""))
        operation_mode = str(st.session_state.get("user_mode", "OPERADOR"))

        xml_path_str = ""
        mes_status = "LOCAL"
        if mes_enabled:
            try:
                xml_path_str = create_inspection_xml(
                    inspection_id=inspection_id,
                    system_name=system_name,
                    equipment_id=equipment_id,
                    mes_enabled=mes_enabled,
                    traceability_enabled=traceability_enabled,
                    production_order=production_order,
                    serial_number=serial_number,
                    model_name=str(st.session_state.get("product_model", "")),
                    line_name=str(st.session_state.get("line_name", "")),
                    operation_mode=operation_mode,
                    result_left=result_left,
                    result_right=result_right,
                    final_result=final_result,
                    confidence_left=conf_left,
                    confidence_right=conf_right,
                    image_path=image_stub,
                    mes_status="PENDENTE",
                    source=trigger_source,
                )
                mes_status = "PENDENTE"
                st.session_state["last_xml_path"] = xml_path_str
            except Exception as xml_err:
                mes_status = f"ERRO_XML: {xml_err}"
                st.session_state["last_xml_path"] = ""
        else:
            st.session_state["last_xml_path"] = ""

        st.session_state["last_mes_status"] = mes_status

        append_trace_log_csv({
            "timestamp": timestamp,
            "inspection_id": inspection_id,
            "system_name": system_name,
            "equipment_id": equipment_id,
            "mes_enabled": mes_enabled,
            "traceability_enabled": traceability_enabled,
            "production_order": production_order,
            "serial_number": serial_number,
            "model_name": str(st.session_state.get("product_model", "")),
            "line": str(st.session_state.get("line_name", "")),
            "operation_mode": operation_mode,
            "source": trigger_source,
            "result_left": result_left,
            "result_right": result_right,
            "final_result": final_result,
            "confidence_left": f"{conf_left:.6f}",
            "confidence_right": f"{conf_right:.6f}",
            "image_path": image_stub,
            "xml_path": xml_path_str,
            "mes_status": mes_status,
        })

        if traceability_enabled:
            st.session_state["serial_qr_code"] = ""

    except Exception as e:
        st.session_state["last_error"] = f"Erro na inferência: {e}"
        st.session_state["last_result"] = None

# ============================================================
#  Lógica de mola faltando ou deslocada
# ============================================================
def detect_missing_spring_simple(roi_bgr, empty_threshold=0.12):
    """
    Heurística simples (placeholder industrial):
    Retorna True se parecer "sem mola".
    Ideal: substituir pelo seu classificador antigo de presença OU uma regra CV melhor.
    """
    import numpy as np
    import cv2

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # normaliza 0..1 e mede "energia de borda" (mola tem muita borda/texture)
    edges = cv2.Canny(gray, 50, 150)
    score = float(np.mean(edges > 0))  # fração de pixels com borda (0..1)

    # se quase não tem borda, suspeita de ROI vazia / sem mola
    return score < empty_threshold, score


def classify_roi_industrial(roi_bgr, use_mnv2, mnv2_model, class_names, pos_idx, thr_ng, img_size):
    """Compat helper com decisão robusta por bandas."""
    is_missing, missing_score = detect_missing_spring_simple(roi_bgr, empty_threshold=0.12)
    if is_missing:
        return {
            "defect_code": "NG_MISSING",
            "missing_score": missing_score,
            "prob_ng": None,
            "prob_ok": None,
            "decision_band": "MISSING",
            "attention_flag": False,
        }

    if use_mnv2:
        _, prob_ng, probs = infer_mobilenetv2_prod(
            roi_bgr, mnv2_model, class_names, pos_idx, float(st.session_state.get("threshold_ng_ng", DEFAULT_THR_NG_NG)), img_size=img_size
        )
        prob_ok = float((probs or {}).get("OK", 1.0 - prob_ng))
        defect_code, decision_band, margin, attention_flag = decide_misaligned_status(
            prob_ng, prob_ok,
            float(st.session_state.get("threshold_ng_ok", DEFAULT_THR_NG_OK)),
            float(st.session_state.get("threshold_ng_ng", DEFAULT_THR_NG_NG)),
            margin_abs=float(st.session_state.get("prod_margin_abs", 0.10)),
        )
        return {
            "defect_code": defect_code,
            "missing_score": missing_score,
            "prob_ng": float(prob_ng),
            "prob_ok": prob_ok,
            "decision_band": decision_band,
            "attention_flag": bool(attention_flag),
            "margin_ok_minus_ng": float(margin),
        }

    return {
        "defect_code": "OK",
        "missing_score": missing_score,
        "prob_ng": None,
        "prob_ok": None,
        "decision_band": "OK_SAFE",
        "attention_flag": False,
    }


def fuse_dual_industrial(left_res, right_res):
    # Prioridade de causa
    if left_res["defect_code"] == "NG_MISSING" or right_res["defect_code"] == "NG_MISSING":
        return "NG_MISSING"
    if left_res["defect_code"] == "NG_MISALIGNED" or right_res["defect_code"] == "NG_MISALIGNED":
        return "NG_MISALIGNED"
    return "OK"

# ==========================================================
# ACTIONS — Capturar + Inferir (BOTÃO + SENSOR)
# ==========================================================

if btn_capture:
    run_capture_infer_dual(trigger_source="button")

# =========================
# SENSOR AUTO JOB
# =========================
# Executa a inspeção armada pelo Serial de forma estável e NÃO-BLOQUEANTE.
# Importante: não usamos time.sleep aqui; aguardamos o tempo de settle via timestamps,
# para evitar que o auto-refresh interrompa a execução e deixe a UI "presa" em firing.
if st.session_state.get("sensor_job_pending", False) and (st.session_state.get("sensor_job_kind") == "sensor"):
    now = time.time()
    settle_ms = int(st.session_state.get("sensor_settle_ms", 220))
    armed_ts = float(st.session_state.get("sensor_job_armed_ts", now))
    ready_at = float(st.session_state.get("sensor_job_ready_at", armed_ts + (settle_ms / 1000.0)))

    # Espera o tempo de estabilização SEM bloquear o script (não depender de sleep)
    if now < ready_at:
        st.session_state["last_sensor_fire_status"] = "arming..."
        st.session_state["last_sensor_fire_error"] = ""
    else:
        # Executa 1x quando estiver pronto
        if not st.session_state.get("capture_busy", False):
            st.session_state["capture_busy"] = True
            st.session_state["capture_busy_since"] = time.time()
            st.session_state["last_sensor_fire_error"] = ""
            st.session_state["last_sensor_fire_status"] = "firing..."
            st.session_state["last_sensor_fire_ts"] = now
            try:
                cap = st.session_state.get("cap")
                cam_ok = bool(st.session_state.get("camera_on", False)) and (cap is not None)

                # Se o preview está ativo via cap, mas camera_on ficou False por algum motivo, normaliza
                if (cap is not None) and (not st.session_state.get("camera_on", False)):
                    try:
                        if hasattr(cap, "isOpened") and cap.isOpened():
                            st.session_state["camera_on"] = True
                            cam_ok = True
                    except Exception:
                        pass

                if cam_ok:
                    print("[SENSOR] trigger solicitado")
                    run_capture_infer_dual(trigger_source="sensor")

                    # conclui job
                    st.session_state["sensor_job_pending"] = False
                    st.session_state["sensor_job_kind"] = None
                    if st.session_state.get("last_result") is None:
                        err = st.session_state.get("last_error") or "Inferência não gerou resultado."
                        st.session_state["last_sensor_fire_error"] = str(err)
                        st.session_state["last_sensor_fire_status"] = "ERR"
                    else:
                        st.session_state["last_sensor_fire_status"] = "OK (infer done)"
                else:
                    st.session_state["last_sensor_fire_error"] = "Trigger ignorado: câmera desligada/indisponível."
                    st.session_state["last_sensor_fire_status"] = "SKIP"
            except Exception as e:
                st.session_state["last_sensor_fire_error"] = str(e)
                st.session_state["last_sensor_fire_status"] = "ERR"
                st.session_state["last_error"] = f"Erro no trigger do sensor: {e}"
            finally:
                st.session_state["capture_busy"] = False
                st.session_state["capture_busy_since"] = 0.0
                # consome a job (1 disparo por armação)
                st.session_state["sensor_job_pending"] = False
                st.session_state["sensor_job_kind"] = None

# ==========================================================
# Frame live (visualização) + assinatura p/ detecção de troca
# ==========================================================
frame = None

# Se estiver em modo congelado (quando existir no app), respeita.
if st.session_state.get("frozen", False) and st.session_state.get("frozen_frame") is not None:
    frame = st.session_state["frozen_frame"].copy()
    st.session_state["live_frame"] = None  # congelado não é "ao vivo"

# Caso normal: sempre tenta ler frame AO VIVO quando a câmera está ON.
elif st.session_state.get("camera_on") and st.session_state.get("cap") is not None:
    frame = read_one_frame(st.session_state["cap"])
    if frame is not None:
        # frame ao vivo disponível (para preview e detecção de troca)
        st.session_state["live_frame"] = frame.copy()
        st.session_state["last_frame"] = frame.copy()
    else:
        st.session_state["live_frame"] = None

# fallback
else:
    st.session_state["live_frame"] = None
    upload_frame = st.session_state.get("upload_test_frame")
    if upload_frame is not None:
        frame = upload_frame.copy()
    else:
        lf = st.session_state.get("last_frame")
        frame = lf.copy() if lf is not None else None

# Assinatura do frame atual (para detectar troca de peça mesmo se PRESENT não voltar a 0)
if frame is not None:
    st.session_state["live_sig"] = quick_frame_signature(frame)


# ==========================================================
# Auto-trigger extra: troca de peça por mudança de imagem (quando sensor fica PRESENT=1 contínuo)
# ==========================================================
ss = st.session_state
if ss.get("serial_on", False) and bool(ss.get("sensor_present", False)) and bool(ss.get("serial_autorefresh", True)):
    # Só tenta armar se não estiver ocupado e não houver job pendente
    if (not ss.get("capture_busy", False)) and (not ss.get("sensor_job_pending", False)):
        now = time.time()
        min_interval = float(ss.get("serial_min_interval_s", 0.8))
        last_ts = float(ss.get("last_infer_ts", 0.0))
        # diferença entre frame atual e último frame que foi inferido
        diff_thr = float(ss.get("serial_image_diff_thr", 6.0))
        live_sig = ss.get("live_sig", None)
        last_sig = ss.get("last_infer_sig", None)
        d = signature_diff(live_sig, last_sig)

        # Rearma por imagem: se mudou bastante e passou um intervalo mínimo, arma uma inspeção nova.
        if (now - last_ts) >= min_interval and d >= diff_thr:
            # Só arma se não houver job pendente (evita re-armar em loop)
            if (not ss.get("sensor_job_pending", False)) and (not ss.get("capture_busy", False)):
                ss["sensor_job_pending"] = True
                ss["sensor_job_kind"] = "sensor"
                ss["sensor_job_armed_ts"] = now
                ss["sensor_job_ready_at"] = now + (float(ss.get("sensor_settle_ms", 220)) / 1000.0)
                ss["last_sensor_fire_status"] = f"arming...(imgΔ={d:.1f})"
            ss["last_sensor_fire_error"] = ""
# ==========================================================
# MAIN — mensagens de erro
# ==========================================================
if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])
if st.session_state.get("last_warning"):
    st.warning(st.session_state["last_warning"])

# ==========================================================
# LAYOUT
# ==========================================================
colA, colB = st.columns([2.0, 1.3], gap="medium")

with colA:
    with st.container(border=True):
        st.markdown("#### Visualização")
        if frame is None:
            st.warning("Sem frame (ligue a câmera ou envie uma imagem no modo Engenharia).")
        else:
            upload_name = str(st.session_state.get("upload_test_name", "")).strip()
            if upload_name and not st.session_state.get("camera_on", False):
                st.caption(f"Imagem de teste carregada: {upload_name}")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=800)

    st.markdown('<div class="compact-divider"></div>', unsafe_allow_html=True)

    

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

            # Resultado Industrial (visual profissional)
            render_resultado_industrial(res)

            if is_eng:
                st.markdown("### 📸 Coleta Manual / Auditoria")

                manual_counts = get_manual_detail_counts()
                last_auto = st.session_state.get("evidence_last_saved")
                last_manual = st.session_state.get("evidence_last_manual")
                last_manual_detail = str(st.session_state.get("manual_last_saved_detail", "")).strip()

                top1, top2 = st.columns([2.2, 1.2])
                with top1:
                    if st.button("✅ Confirmar OK (ambas perfeitas)", use_container_width=True):
                        try:
                            saved = save_manual_current_result(detail_override="OK")
                            st.success(f"Salvo: {manual_detail_human('OK')}")
                        except Exception as e:
                            st.error(f"Falha ao salvar OK: {e}")
                with top2:
                    st.metric("Contador OK", manual_counts["OK"])

                st.markdown("**NG_DESALINHADO**")
                d1, d2, d3 = st.columns(3)
                with d1:
                    if st.button("⚠️ ESQ desalinhado / DIR OK", key="btn_des_esq", use_container_width=True):
                        try:
                            save_manual_current_result(detail_override="DESALINHADA_ESQ")
                            st.warning("Salvo: Desalinhado ESQ")
                        except Exception as e:
                            st.error(f"Falha ao salvar: {e}")
                    st.caption(f"Contador: {manual_counts['DESALINHADA_ESQ']}")
                with d2:
                    if st.button("⚠️ ESQ OK / DIR desalinhado", key="btn_des_dir", use_container_width=True):
                        try:
                            save_manual_current_result(detail_override="DESALINHADA_DIR")
                            st.warning("Salvo: Desalinhado DIR")
                        except Exception as e:
                            st.error(f"Falha ao salvar: {e}")
                    st.caption(f"Contador: {manual_counts['DESALINHADA_DIR']}")
                with d3:
                    if st.button("⚠️ Ambos desalinhados", key="btn_des_both", use_container_width=True):
                        try:
                            save_manual_current_result(detail_override="DESALINHADA_BOTH")
                            st.warning("Salvo: Desalinhado ambos")
                        except Exception as e:
                            st.error(f"Falha ao salvar: {e}")
                    if manual_counts["DESALINHADA_BOTH"] > 0:
                        st.info(f"Salvo: Desalinhado ambos")
                    st.caption(f"Contador: {manual_counts['DESALINHADA_BOTH']}")

                st.markdown("**NG_FALTANDO**")
                f1, f2, f3 = st.columns(3)
                with f1:
                    if st.button("❌ ESQ faltando / DIR OK", key="btn_falt_esq", use_container_width=True):
                        try:
                            save_manual_current_result(detail_override="FALTANDO_ESQ")
                            st.error("Salvo: Faltando ESQ")
                        except Exception as e:
                            st.error(f"Falha ao salvar: {e}")
                    st.caption(f"Contador: {manual_counts['FALTANDO_ESQ']}")
                with f2:
                    if st.button("❌ ESQ OK / DIR faltando", key="btn_falt_dir", use_container_width=True):
                        try:
                            save_manual_current_result(detail_override="FALTANDO_DIR")
                            st.error("Salvo: Faltando DIR")
                        except Exception as e:
                            st.error(f"Falha ao salvar: {e}")
                    st.caption(f"Contador: {manual_counts['FALTANDO_DIR']}")
                with f3:
                    if st.button("❌ Ambas faltando", key="btn_falt_both", use_container_width=True):
                        try:
                            save_manual_current_result(detail_override="FALTANDO_BOTH")
                            st.error("Salvo: Faltando ambos")
                        except Exception as e:
                            st.error(f"Falha ao salvar: {e}")
                    st.caption(f"Contador: {manual_counts['FALTANDO_BOTH']}")

                st.markdown("**Casos mistos (um defeito em cada lado)**")
                m1, m2 = st.columns(2)
                with m1:
                    if st.button("🔀 ESQ desalinhado / DIR faltando", key="btn_misto_des_falt", use_container_width=True):
                        try:
                            save_manual_current_result(detail_override="MISTO_DESALINHADA_ESQ_FALTANDO_DIR")
                            st.info("Salvo: ESQ desalinhado / DIR faltando")
                        except Exception as e:
                            st.error(f"Falha ao salvar: {e}")
                    st.caption(f"Contador: {manual_counts['MISTO_DESALINHADA_ESQ_FALTANDO_DIR']}")
                with m2:
                    if st.button("🔀 ESQ faltando / DIR desalinhado", key="btn_misto_falt_des", use_container_width=True):
                        try:
                            save_manual_current_result(detail_override="MISTO_FALTANDO_ESQ_DESALINHADA_DIR")
                            st.info("Salvo: ESQ faltando / DIR desalinhado")
                        except Exception as e:
                            st.error(f"Falha ao salvar: {e}")
                    st.caption(f"Contador: {manual_counts['MISTO_FALTANDO_ESQ_DESALINHADA_DIR']}")

                if last_auto:
                    st.info(f"Última evidência automática: **{last_auto['raw_path'].name}** — {last_auto['reason']}")
                if last_manual:
                    detail_txt = manual_detail_human(last_manual_detail) if last_manual_detail else last_manual['label']
                    st.info(f"Último salvamento manual: **{detail_txt}** — `{last_manual['raw_path'].name}`")

                with st.expander("🗂 Monitor da pasta automática", expanded=False):
                    auto_bytes = folder_size_bytes(AUTO_EVIDENCE_DIR)
                    auto_files = count_evidence_files(AUTO_EVIDENCE_DIR)
                    disk_info = get_disk_status(BASE_DIR)
                    disk_status, disk_label = disk_free_status_label(disk_info.get("free_gb", 0.0), warn_gb=10.0, critical_gb=5.0)
                    st.write(f"**Pasta:** `{AUTO_EVIDENCE_DIR}`")
                    st.write(f"**Tamanho atual:** {bytes_to_human(auto_bytes)}")
                    st.write(f"**Arquivos atuais:** {auto_files}")
                    st.write(f"**Espaço livre em disco:** {disk_info.get('free_gb', 0.0):.2f} GB")
                    st.write(f"**Status do disco:** {disk_label}")
                    st.write(f"**Retenção:** {int(st.session_state.get('evidence_retention_days', 60))} dias" if bool(st.session_state.get('evidence_retention_enabled', True)) else "**Retenção:** desativada")
                    recent = list_recent_files(AUTO_EVIDENCE_DIR, limit=8)
                    if recent:
                        st.caption("Últimos arquivos:")
                        for p in recent:
                            st.write(f"- {p.name}")

        if len(st.session_state.get("history", [])) > 1:
            with st.expander("📈 Qualidade (Gráficos)", expanded=False):
                import pandas as pd
                df = pd.DataFrame(st.session_state.get("history", []))

                # Compatibilidade com históricos antigos ou entradas vindas do upload.
                # Garante que o painel não quebre se alguma linha não tiver a chave
                # "aprovado" ou "n" no formato esperado.
                if "aprovado" not in df.columns:
                    if "result" in df.columns:
                        df["aprovado"] = df["result"].astype(str).str.upper().eq("OK").astype(int)
                    elif "defect_type" in df.columns:
                        df["aprovado"] = df["defect_type"].astype(str).str.upper().eq("OK").astype(int)
                    else:
                        df["aprovado"] = 0
                else:
                    df["aprovado"] = pd.to_numeric(df["aprovado"], errors="coerce").fillna(0).astype(int)

                if "n" not in df.columns:
                    df["n"] = range(1, len(df) + 1)
                else:
                    df["n"] = pd.to_numeric(df["n"], errors="coerce")
                    if df["n"].isna().any():
                        df["n"] = range(1, len(df) + 1)
                    df["n"] = df["n"].astype(int)

                df["ok_cum"] = df["aprovado"].cumsum()
                df["yield_cum"] = 100.0 * df["ok_cum"] / df["n"].clip(lower=1)

                st.caption("Tendência de Yield (%)")
                st.line_chart(df.set_index("n")[["yield_cum"]])

                st.caption("Defeitos por lado (NG)")
                defects = {
                    "ESQ": int(st.session_state.get("cnt_ng_esq", 0)),
                    "DIR": int(st.session_state.get("cnt_ng_dir", 0))
                }
                st.bar_chart(defects)

        audit_counts = get_audit_counts_from_session()
        with st.expander("🧾 Painel detalhado de auditoria", expanded=True):
            a1, a2, a3 = st.columns(3)
            with a1:
                st.metric("Faltando ESQ", audit_counts["faltando_esq"])
                st.metric("Faltando DIR", audit_counts["faltando_dir"])
                st.metric("Faltando BOTH", audit_counts["faltando_both"])
            with a2:
                st.metric("Desalinhada ESQ", audit_counts["desalinhada_esq"])
                st.metric("Desalinhada DIR", audit_counts["desalinhada_dir"])
                st.metric("Desalinhada BOTH", audit_counts["desalinhada_both"])
            with a3:
                st.metric("Casos mistos", audit_counts["misto"])
                st.metric("OK com atenção", audit_counts["ok_atencao"])
                st.metric("NG total", int(st.session_state.get("cnt_ng", 0)))


        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        btn_col, info_col = st.columns([1.1, 1.2])
        with btn_col:
            if st.button("📄 Gerar Relatório de Auditoria", key="btn_generate_audit_report", use_container_width=True):
                try:
                    pdf_path, html_path = generate_audit_report_files()
                    st.success(f"Relatório gerado com sucesso: {pdf_path.name}")
                    if bool(st.session_state.get("email_reports_enabled", False)) and bool(st.session_state.get("email_send_on_generate", False)):
                        ok_email, msg_email = send_report_email(pdf_path, html_path)
                        if ok_email:
                            st.success(f"📧 {msg_email}")
                        else:
                            st.error(f"Falha ao enviar relatório por e-mail: {msg_email}")
                except Exception as e:
                    st.error(f"Falha ao gerar relatório: {e}")
        with info_col:
            if st.session_state.get("last_report_generated_at"):
                st.caption(f"Último relatório: {st.session_state.get('last_report_generated_at')}")

        last_pdf = st.session_state.get("last_report_pdf", "")
        last_html = st.session_state.get("last_report_html", "")
        if last_pdf and Path(last_pdf).exists():
            d1, d2, d3 = st.columns(3)
            with d1:
                with open(last_pdf, "rb") as fpdf:
                    st.download_button(
                        "⬇️ Baixar PDF do relatório",
                        data=fpdf.read(),
                        file_name=Path(last_pdf).name,
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_pdf_report",
                    )
            with d2:
                if last_html and Path(last_html).exists():
                    with open(last_html, "rb") as fhtml:
                        st.download_button(
                            "⬇️ Baixar HTML resumo",
                            data=fhtml.read(),
                            file_name=Path(last_html).name,
                            mime="text/html",
                            use_container_width=True,
                            key="download_html_report",
                        )
        ####with d3:############################ Retirada do botão Enviar relatório por e-mail 
        #       if st.button("📧 Enviar relatório por e-mail", key="btn_send_report_email", use_container_width=True):
        #           try:
        #               ok_email, msg_email = send_report_email(last_pdf, last_html)
        #               if ok_email:
        #                   st.success(msg_email)
        #               else:
        #                   st.error(f"Falha ao enviar relatório por e-mail: {msg_email}")
        #           except Exception as e:
        #               st.error(f"Falha ao enviar relatório por e-mail: {e}")

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

# ==========================================================
# CLEANUP — garante liberação da porta serial
# ==========================================================
import atexit

def _cleanup_serial():
    try:
        serial_stop()
    except Exception:
        pass

atexit.register(_cleanup_serial)

# Executa o job do sensor no final do script (após todas as defs)
execute_sensor_job_if_ready()


# ==========================================================
# EXEC SENSOR JOB (determinístico):
# quando o Serial arma `sensor_job_pending`, roda captura+infer
# ==========================================================
_ss = st.session_state
if _ss.get("serial_on", False) and _ss.get("sensor_job_pending", False) and not _ss.get("capture_busy", False):
    _now = time.time()
    _ready_at = float(_ss.get("sensor_job_ready_at", 0.0))
    if _now >= _ready_at:
        _ss["capture_busy"] = True
        _ss["capture_busy_since"] = _now
        try:
            ensure_active_model_loaded_or_raise(blocking=True)
            run_capture_infer_dual(trigger_source="sensor")
            _ss["last_error"] = None
            _ss["last_sensor_fire_status"] = "OK (infer done)"
            _ss["last_sensor_fire_error"] = ""
        except Exception as _e:
            _ss["last_sensor_fire_status"] = "ERR"
            _ss["last_sensor_fire_error"] = str(_e)
            _ss["last_error"] = f"Erro na inferência: {_e}"
        finally:
            _ss["sensor_job_pending"] = False
            _ss["sensor_job_kind"] = None
            _ss["capture_busy"] = False
            _ss["capture_busy_since"] = 0.0


# ==========================================================
# ANDON - Alerta automático de Yield baixo (não altera lógica existente)
# ==========================================================
def check_andon_alert():
    try:
        total = int(st.session_state.get("cnt_total", 0))
        ok = int(st.session_state.get("cnt_ok", 0))

        if total == 0:
            return

        yield_pct = (ok / total) * 100.0

        MIN_PRODUCTION = 100
        YIELD_THRESHOLD = 95.0

        if total >= MIN_PRODUCTION and yield_pct < YIELD_THRESHOLD:

            st.markdown(f"""
<div class="andon-banner">
🚨 ANDON ACIONADO — YIELD BAIXO<br>
Yield atual: {yield_pct:.2f}% | Produção: {total} peças
</div>
""", unsafe_allow_html=True)

            st.warning("Verificar processo imediatamente.")

            # ------------------------------------------------
            # Disparo automático de email (somente uma vez)
            # ------------------------------------------------
            if not st.session_state.get("andon_email_sent", False):

                try:
                    last_pdf = st.session_state.get("last_pdf_report")
                    last_html = st.session_state.get("last_html_report")

                    # Só envia se existir relatório
                    if last_pdf and last_html:

                        ok_email, msg_email = send_report_email(last_pdf, last_html)

                        if ok_email:
                            st.session_state["andon_email_sent"] = True
                            st.warning("📧 Email de ALERTA ANDON enviado automaticamente.")

                    else:
                        st.warning("⚠ ANDON acionado, mas relatório ainda não foi gerado.")

                except Exception as e:
                    st.error(f"Erro ao enviar email ANDON: {e}")

        # ------------------------------------------------
        # Reset automático quando processo normaliza
        # ------------------------------------------------
        elif yield_pct >= YIELD_THRESHOLD:
            st.session_state["andon_email_sent"] = False

    except Exception:
        pass


# ==========================================================
# CHAMADA DO ANDON (não interfere em outras funções)
# ==========================================================
try:
    check_andon_alert()
except Exception:
    pass