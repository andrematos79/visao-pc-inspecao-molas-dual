
# SVC — Computer Vision System for Spring Inspection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19207171.svg)](https://doi.org/10.5281/zenodo.19207171)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Release](https://img.shields.io/badge/version-v1.0.0--industrial-orange)

Latest version DOI: https://doi.org/10.5281/zenodo.19207171
Concept DOI (all versions): https://doi.org/10.5281/zenodo.19207170

Industrial computer vision system for automated inspection of springs in
mobile phone chargers.

This repository contains the validated industrial version of the **SVC
(Spring Vision Control)** system used for automated quality inspection
in production environments.

-----------------------------------------------------------------------

# Overview

This project presents a **low‑cost computer vision system for automated
inspection of metallic springs installed inside mobile phone chargers**.

The system combines:

- USB camera image acquisition
- Convolutional Neural Networks (MobileNetV2)
- Dual‑ROI industrial decision logic
- Real‑time operator interface
- Industrial traceability logging
- Automated evidence storage for auditing
- Automated reporting and email notifications

The objective is to automatically detect **spring presence and alignment
defects** inside the charger housing.

The solution was developed within the **Professional Master's Program in
Electrical Engineering — Embedded Systems and Computer Vision** at the
**Universidade do Estado do Amazonas (UEA)**.

Developed at the **Embedded Systems and Computer Vision Laboratories**
of the **School of Technology (EST)**.

-----------------------------------------------------------------------

# Key Features

✔ Low‑cost industrial computer vision system  
✔ CNN‑based inspection using MobileNetV2  
✔ Dual‑ROI inspection strategy  
✔ CPU‑only inference (no GPU required)  
✔ Automatic triggering via proximity sensor  
✔ Industrial operator interface using Streamlit  

### Advanced Industrial Features

✔ Automatic dataset collection for retraining  
✔ Automatic saving of NG (defective) evidence images  
✔ Disk usage monitoring for evidence storage folder  
✔ Automatic retention policies for stored images (30 / 60 / 90 days)  
✔ Integrated audit inspection panel on the interface  
✔ Automatic generation of inspection reports (immediate, shift and daily)  
✔ Email delivery of inspection reports to engineering or management  

-----------------------------------------------------------------------

# Automated Dataset Collection

The system includes an integrated **dataset collection module** that
allows automatic capture of inspection images during production.

This feature enables:

- Continuous dataset expansion
- Collection of real production samples
- Faster retraining of AI models
- Improved defect detection accuracy

Images are automatically organized into dataset folders for future
machine learning training.

-----------------------------------------------------------------------

# Evidence Management System

When a defect is detected, the system automatically saves an **NG
evidence image**.

These images support:

- Production quality auditing
- Failure investigation
- Dataset expansion
- Process improvement

The system also includes automated **storage management**:

- Disk space monitoring
- Automatic storage alerts
- Configurable image retention policies

Retention policies supported:

- 30 days
- 60 days
- 90 days

Older images are automatically removed according to the configured
policy.

-----------------------------------------------------------------------

# Audit Interface

The SVC system includes a built‑in **audit visualization panel**
accessible directly from the operator interface.

The audit panel allows engineers and auditors to visualize:

- Inspection results
- Detected defect type
- ROI classification results
- Confidence scores
- Inspection timestamps
- Production statistics

This eliminates the need to manually open image folders during audits.

-----------------------------------------------------------------------

# Automated Reporting System

The system can automatically generate inspection reports.

Supported report types:

- Immediate inspection reports
- Shift reports
- Daily production reports

Reports include:

- Inspection statistics
- Defect distribution
- Production yield
- Traceability information

Reports are automatically stored for future auditing.

-----------------------------------------------------------------------

# Email Notification System

Inspection reports can be automatically sent via email to:

- Quality engineers
- Production managers
- Process engineers
- Industrial auditors

This allows **remote monitoring of production quality indicators**.

-----------------------------------------------------------------------

# System Architecture

Operational pipeline:

Sensor Trigger  
↓  
Image Acquisition  
↓  
ROI Extraction (Left / Right spring regions)  
↓  
CNN Classification (MobileNetV2)  
↓  
Dual Decision Logic  
↓  
Operator Interface + Production Logging  
↓  
Evidence Storage + Reporting + Email Notification

The system also supports **automatic triggering using a proximity sensor
connected to an Arduino Uno**.

-----------------------------------------------------------------------

# Hardware Components

Computer: Industrial PC (Windows 10 / 11)  
CPU: Intel Core i3 12th Gen or higher  
RAM: 8 GB or more  
Camera: Industrial USB camera  
Microcontroller: Arduino Uno  
Sensor: E18‑D80NK proximity sensor

-----------------------------------------------------------------------

# Software Stack

Python  
TensorFlow / Keras  
OpenCV  
Streamlit  
PySerial  
Pandas  
Matplotlib  

-----------------------------------------------------------------------

# Artificial Intelligence Model

The inspection system uses **MobileNetV2 with Transfer Learning**.

### Model Classes

OK — Spring correctly installed  
NG_MISSING — Spring absent  
NG_MISALIGNED — Spring present but misaligned

Each spring position is evaluated independently.

-----------------------------------------------------------------------

# Dual Decision Logic

A product is approved only if:

ROI Left = OK  
AND  
ROI Right = OK

If any side presents a defect, the product is rejected.

-----------------------------------------------------------------------

# Industrial Validation

Dataset: ~1170 real images collected from production environment.

Validation experiment:

OK — 50 units  
NG_MISALIGNED — 30 units  
NG_MISSING — 20 units

Average inspection time:

**1.93 seconds per unit**

-----------------------------------------------------------------------

# Installation

Create project folder:

C:\SVC_INSPECAO_MOLAS

Create virtual environment:

python -m venv .venv_svc

Activate environment:

.\.venv_svc\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

-----------------------------------------------------------------------

# Running the System

streamlit run app_camera_infer_dual_freeze.py

-----------------------------------------------------------------------

# Research Context

This software was developed as part of research on **automated industrial
inspection using computer vision and deep learning**.

The project contributes to **Smart Manufacturing and Industry 4.0** by
providing a low‑cost automated inspection solution for electronic
manufacturing lines.

The system evolved from a **laboratory prototype into an industrially
validated solution**, increasing its Technology Readiness Level (TRL).

-----------------------------------------------------------------------

# How to Cite

Matos, A. G. (2026)  
**SVC — Computer Vision System for Spring Inspection**  
Zenodo  
https://doi.org/10.5281/zenodo.19207170

-----------------------------------------------------------------------

# Author

André Gama de Matos  
Engineer — Computer Vision and Embedded Systems

Professional Master's in Electrical Engineering  
Universidade do Estado do Amazonas (UEA)

Advisor:
Prof. Dr. Carlos Maurício Seródio Figueiredo

Co‑Advisor:
Prof. Dr. Jozias Parente de Oliveira

-----------------------------------------------------------------------

# License

MIT License — Open source software for research and industrial
experimentation.
