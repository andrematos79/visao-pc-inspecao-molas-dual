
# SVC — Computer Vision System for Automated Spring Inspection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19323586.svg)](https://doi.org/10.5281/zenodo.19323586)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Release](https://img.shields.io/badge/version-v1.0.1--industrial-blue)

**Latest version:** **SVC v1.1.0 – Industrial ROI calibration and temporal inference smoothing**
**Latest version DOI:** https://doi.org/10.5281/zenodo.19323586  
**Concept DOI (all versions):** https://doi.org/10.5281/zenodo.19207170

---

# Overview

**SVC (Spring Vision Control)** is a low‑cost industrial computer vision system designed for **automated inspection of metallic springs installed inside mobile phone chargers**.

The system uses **deep learning, industrial triggering mechanisms, and a dual‑ROI inspection strategy** to detect:

• Missing springs  
• Misaligned springs  
• Correct assembly conditions  

The system evolved from a **laboratory prototype to an industrial-ready solution (v1.1.0)** validated and calibrated under real manufacturing conditions.

This work was developed within the **Professional Master's Program in Electrical Engineering (PPGEEL)** at the **Universidade do Estado do Amazonas (UEA)**, focusing on **Embedded Systems and Computer Vision for Smart Manufacturing applications**.

---

# Key Features

✔ Low‑cost industrial computer vision architecture  
✔ CNN‑based inspection using **MobileNetV2**  
✔ Dual‑ROI inspection strategy for left/right spring regions  
✔ CPU‑only inference (no GPU required)  
✔ Automatic inspection triggering via proximity sensor  
✔ Industrial operator interface built with **Streamlit**  

### Advanced Industrial Features

✔ Automatic dataset collection for AI retraining  
✔ Automatic saving of NG (defective) evidence images  
✔ Disk usage monitoring for evidence storage folders  
✔ Automatic evidence retention policies (30 / 60 / 90 days)  
✔ Integrated audit inspection panel  
✔ Detailed engineering defect analysis panel  
✔ Automatic inspection report generation (immediate / shift / daily)  
✔ Email delivery of inspection reports  
✔ Persistent storage of email configuration settings  
✔ Laboratory inspection mode via **image upload**

---

# System Architecture

Operational pipeline:

Sensor Trigger  
↓  
Image Acquisition (USB Industrial Camera)  
↓  
ROI Extraction (Left / Right Spring Regions)  
↓  
CNN Classification (MobileNetV2)  
↓  
Dual Industrial Decision Logic  
↓  
Operator Interface + Production Logging  
↓  
Evidence Storage + Reporting + Email Notification

The system supports **automatic triggering using a proximity sensor connected to an Arduino Uno microcontroller**.

---

# Hardware Components

Industrial PC (Windows 10 / 11)  
Intel Core i3 12th Gen or higher  
8 GB RAM minimum  
Industrial USB camera  
Arduino Uno microcontroller  
E18‑D80NK proximity sensor

---

# Software Stack

Python  
TensorFlow / Keras  
OpenCV  
Streamlit  
PySerial  
Pandas  
Matplotlib  

---

# Artificial Intelligence Model

The inspection system uses **MobileNetV2 with Transfer Learning**.

### Model Classes

OK — Spring correctly installed  
NG_MISSING — Spring absent  
NG_MISALIGNED — Spring present but misaligned

Each spring location is analyzed independently.

---

# Dual Decision Logic

A product is approved only if:

ROI Left = OK  
AND  
ROI Right = OK

If any ROI detects a defect, the product is automatically rejected.

This strategy improves **industrial robustness and reduces false approvals**.

---

# Dataset Collection System

The SVC includes a built‑in **dataset generation tool** allowing engineers to capture inspection images directly from production.

Benefits:

• Continuous dataset expansion  
• Faster AI retraining cycles  
• Real industrial defect collection  
• Improved model robustness

Images are automatically organized into structured dataset folders.

---

# Evidence Management System

When a defect is detected, the system automatically stores an **NG evidence image**.

These images support:

• Quality audits  
• Failure investigations  
• Dataset expansion  
• Manufacturing process improvements

Storage management features include:

• Disk space monitoring  
• Automatic alerts  
• Configurable retention policies

Retention options:

30 days  
60 days  
90 days

Older evidence images are automatically removed.

---

# Audit Interface

The system provides a built‑in **audit visualization panel** integrated into the operator interface.

Engineers and auditors can inspect:

• Inspection results  
• Detected defect type  
• ROI classification outputs  
• Confidence scores  
• Inspection timestamps  
• Production statistics

This eliminates the need to manually browse evidence folders during industrial audits.

---

# Automated Reporting

The SVC system automatically generates inspection reports.

Supported report types:

Immediate inspection reports  
Shift reports  
Daily production reports

Reports include:

• Production yield  
• Defect distribution  
• Inspection statistics  
• Traceability data

Reports are automatically archived for auditing.

---

# Email Notification System

Inspection reports can be automatically delivered via email to:

Quality engineers  
Production managers  
Process engineers  
Industrial auditors

Email configuration is **persistently stored**, ensuring system continuity after shutdown or restart.

This enables **remote monitoring of production quality indicators**.

---

# Industrial Validation

Dataset size: ~1170 production images.

Experimental validation:

OK — 50 units  
NG_MISALIGNED — 30 units  
NG_MISSING — 20 units

Average inspection time:

**1.93 seconds per unit**

The results demonstrate the feasibility of **low‑cost deep learning systems for industrial inspection tasks**.

---

# Installation

Create project directory:

C:\SVC_INSPECAO_MOLAS

Create virtual environment:

python -m venv .venv_svc

Activate environment:

.\.venv_svc\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

---

# Running the System

streamlit run app_camera_infer_dual_freeze.py

---

# Research Context

This system contributes to research in:

• Industrial computer vision  
• Automated quality inspection  
• Deep learning for manufacturing  
• Smart Manufacturing / Industry 4.0

The project demonstrates the feasibility of **deploying deep learning‑based inspection systems using low‑cost hardware in real manufacturing environments**.

---

# Citation

If you use this system in research or industrial projects, please cite:

Matos, A. G. (2026)  
**SVC — Computer Vision System for Spring Inspection**  
Version 1.0.1  
Zenodo  
https://doi.org/10.5281/zenodo.19323586

---

# Author

André Gama de Matos  
Engineer — Computer Vision and Embedded Systems

Professional Master's in Electrical Engineering  
Universidade do Estado do Amazonas (UEA)

Advisor  
Prof. Dr. Carlos Maurício Seródio Figueiredo

Co‑Advisor  
Prof. Dr. Jozias Parente de Oliveira

---

# License

MIT License — Open source software for research and industrial experimentation.
