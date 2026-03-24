# SVC -- Computer Vision System for Spring Inspection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19207171.svg)](https://doi.org/10.5281/zenodo.19207171)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Release](https://img.shields.io/badge/version-v1.0.1--industrial-orange)

Latest version DOI: https://doi.org/10.5281/zenodo.19207171\
Concept DOI (all versions): https://doi.org/10.5281/zenodo.19207170

Industrial computer vision system for automated inspection of springs in
mobile phone chargers.

This repository contains the validated industrial version of the **SVC
(Spring Vision Control)** system used for automated quality inspection
in production environments.

------------------------------------------------------------------------

# Overview

This project presents a **low‑cost computer vision system for automated
inspection of metallic springs installed inside mobile phone chargers**.

The system combines:

-   USB camera image acquisition
-   Convolutional Neural Networks (MobileNetV2)
-   Dual‑ROI industrial decision logic
-   Real‑time operator interface
-   Industrial traceability logging

The objective is to automatically detect **spring presence and alignment
defects** inside the charger housing.

The solution was developed within the **Professional Master's Program in
Electrical Engineering -- Embedded Systems** at the **Universidade do
Estado do Amazonas (UEA)**.

The system operates in **real industrial environments using CPU‑only
inference**, integrating hardware and software for automated inline
inspection.

This repository represents the **validated industrial version deployed
on the production floor**, evolving from a laboratory prototype into a
production‑ready system.

------------------------------------------------------------------------

# Key Features

✔ Low‑cost industrial computer vision system\
✔ CNN‑based inspection using MobileNetV2\
✔ Dual‑ROI inspection strategy\
✔ CPU‑only inference (no GPU required)\
✔ Automatic triggering via proximity sensor\
✔ Industrial operator interface using Streamlit\
✔ Automatic dataset collection module\
✔ Evidence logging for quality auditing\
✔ Disk usage monitoring and retention policies\
✔ Validated in real industrial environment

------------------------------------------------------------------------

# System Architecture

The SVC system follows a modular architecture designed for industrial
reliability.

Operational pipeline:

USB Camera\
↓\
Image Acquisition\
↓\
ROI Extraction (Left / Right spring regions)\
↓\
CNN Classification (MobileNetV2)\
↓\
Dual Decision Logic\
↓\
Operator Interface + Production Logging

The system also supports **automatic triggering using a proximity sensor
connected to an Arduino Uno**.

------------------------------------------------------------------------

# Hardware Components

  Component         Description
  ----------------- ----------------------------------
  Computer          Windows 11 Pro
  CPU               Intel Core i3 12th Gen or higher
  RAM               8 GB or more
  Camera            Industrial USB camera
  Microcontroller   Arduino Uno
  Sensor            E18‑D80NK proximity sensor
  Interface         USB

The sensor automatically triggers inspection when the product is
positioned in front of the camera.

------------------------------------------------------------------------

# Software Stack

  Software                Purpose
  ----------------------- ---------------------------
  Python                  Main programming language
  TensorFlow / Keras      CNN inference
  OpenCV                  Image processing
  Streamlit               Operator interface
  PySerial                Arduino communication
  Pandas                  Production logging
  Matplotlib              Visualization
  Streamlit Autorefresh   Sensor monitoring

------------------------------------------------------------------------

# Artificial Intelligence Model

The inspection system uses **MobileNetV2 with Transfer Learning** for
defect classification.

### Model Classes

  Class           Description
  --------------- -------------------------------
  OK              Spring correctly installed
  NG_MISSING      Spring absent
  NG_MISALIGNED   Spring present but misaligned

Each spring position is evaluated independently.

------------------------------------------------------------------------

# Dual Decision Logic

The final inspection decision follows a conservative industrial rule.

A product is approved only if:

ROI Left = OK\
AND\
ROI Right = OK

If any side presents a defect, the product is rejected.

This approach increases inspection reliability and reduces false
positives.

------------------------------------------------------------------------

# Industrial Validation

The system was validated using **real products from the production
line**.

### Dataset

The training dataset contains approximately **1170 real images**
collected directly from the industrial inspection setup.

### Experimental Test

A validation test was conducted with **100 real charger units**:

  Class           Quantity
  --------------- ----------
  OK              50
  NG_MISALIGNED   30
  NG_MISSING      20

The system performed:

-   automatic image acquisition
-   ROI extraction
-   CNN inference
-   dual‑logic decision
-   production logging

### Performance

Average processing time:

1.93 seconds per unit

The system demonstrated reliable performance under real industrial
conditions.

------------------------------------------------------------------------

# Project Structure

    SVC_INSPECAO_MOLAS
    │
    ├── app_camera_infer_dual_freeze.py
    ├── config_molas.json
    ├── labels.json
    ├── models_registry.json
    ├── requirements.txt
    │
    ├── assets
    │   └── logo_empresa.jpg
    │
    ├── configs
    │   └── UNICORN_WHITE_15W.json

------------------------------------------------------------------------

# Installation

Create project folder:

    C:\SVC_INSPECAO_MOLAS

Create virtual environment:

    python -m venv .venv_svc

Activate environment:

    .\.venv_svc\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

# Running the System

    streamlit run app_camera_infer_dual_freeze.py

The operator interface will open automatically in the browser.

------------------------------------------------------------------------

# Sensor Connection

  Sensor Wire   Arduino
  ------------- ---------------
  Brown         5V
  Blue          GND
  Black         Digital Pin 2

Serial communication: **115200 baud**

------------------------------------------------------------------------

# Research Context

This software was developed as part of research on **automated
industrial inspection using computer vision and deep learning**.

The project contributes to **smart manufacturing and Industry 4.0**,
providing a low‑cost automated inspection solution for electronic
manufacturing lines.

The system evolved from a **laboratory prototype to an industrially
validated solution**, increasing its Technology Readiness Level (TRL).

------------------------------------------------------------------------

# How to Cite

If you use this software in academic work, please cite:

Matos, A. G. (2026).\
**SVC -- Computer Vision System for Spring Inspection.**\
Zenodo. https://doi.org/10.5281/zenodo.19207170

------------------------------------------------------------------------

# Author

**André Gama de Matos**

Engineer -- Computer Vision and Embedded Systems

Professional Master's in Electrical Engineering -- Embedded Systems\
Universidade do Estado do Amazonas (UEA)

Advisor:\
Prof. Dr. Carlos Maurício Seródio Figueiredo

Co‑Advisor:\
Prof. Dr. Jozias Parente de Oliveira

------------------------------------------------------------------------

# License

MIT License -- Open source software for research and industrial
experimentation.
