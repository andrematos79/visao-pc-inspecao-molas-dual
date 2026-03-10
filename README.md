# Industrial Computer Vision System for Automatic Inspection of Charger Springs

## Overview

This project presents a **low-cost industrial computer vision system** designed for automatic inspection of metallic springs installed inside charger covers.  

The system uses a **USB camera**, a **convolutional neural network (MobileNetV2)** and a **dual-region decision logic (DUAL)** to classify the presence and alignment of springs in two independent regions of interest (ROIs).

The solution was developed as part of a **Professional Master's Degree in Electrical Engineering – Embedded Systems** at the **Universidade do Estado do Amazonas (UEA)**.

The system operates in real industrial environments using **CPU-only inference**, integrating hardware and software components for automated inspection.

---

# System Architecture

The SVC integrates multiple components:

USB Camera
↓
Image Capture
↓
ROI Extraction (Left / Right)
↓
MobileNetV2 CNN Classification
↓
Dual Decision Logic
↓
Result Visualization + Production Logging


Additionally, the system integrates **automatic triggering via a proximity sensor connected to an Arduino Uno**.

## System Architecture pictures

![System Architecture](docs/figures/system_architecture.png)

## Operator Interface

![System Interface](docs/figures/interface_operator.png)

## Prototype

![Prototype](docs/figures/prototype_system.png)

---

# Hardware Components

The following hardware configuration was used in the validated system:

| Component | Description |
|--------|--------|
| Computer | Windows 11 Pro |
| CPU | Intel Core i3 12th Gen or higher |
| RAM | 8 GB or more |
| Camera | USB industrial camera |
| Microcontroller | Arduino Uno |
| Sensor | E18-D80NK proximity sensor |
| Interface | USB |

The sensor automatically triggers inspection when a product is positioned in front of the camera.

---

# Software Stack

The system was developed using Python and modern computer vision and machine learning libraries.

| Software | Function |
|--------|--------|
| Python | Main programming language |
| TensorFlow / Keras | CNN inference |
| OpenCV | Image processing |
| Streamlit | Operator interface |
| PySerial | Arduino communication |
| Pandas | Production log handling |
| Matplotlib | Visualization |
| Streamlit Autorefresh | Automatic sensor polling |

---

# AI Model

The system uses **MobileNetV2 with transfer learning** to classify the condition of the springs.

### Classes

The neural network performs multiclass classification for each ROI:

| Class | Description |
|------|------|
| OK | Spring correctly installed |
| NG_MISSING | Spring missing |
| NG_MISALIGNED | Spring present but misaligned |

Each side of the product is evaluated independently.

---

# Dual Decision Logic

The final product decision follows a conservative industrial rule:

Product = OK
only if
Left ROI = OK
AND
Right ROI = OK


If any side is classified as **NG**, the product is rejected.

This strategy increases reliability in production environments.

---

# System Modes

The system can operate in two modes.

### Manual Mode

The operator triggers inspection manually.

### Automatic Mode

Inspection is triggered automatically by the **E18-D80NK proximity sensor** connected to the Arduino.

The Arduino sends serial messages:

PRESENT=0
PRESENT=1


The Streamlit interface monitors the serial port and triggers inference.

---

# Project Structure

SVC_INSPECAO_MOLAS
│
├── app_camera_infer_dual_freeze.py
├── modelo_molas.keras
├── config_molas.json
├── production_log.csv
├── requirements.txt
├── requirements_lock_prod.txt
│
├── assets
│ └── logo_empresa.jpg
│
└── .venv_svc


---

# Installation

### 1. Create project directory

C:\SVC_INSPECAO_MOLAS


### 2. Create virtual environment

python -m venv .venv_svc


### 3. Activate environment

..venv_svc\Scripts\Activate.ps1

### 4. Install dependencies

pip install -r requirements.txt

Or replicate the validated environment: pip install -r requirements_lock_prod.txt


---

# Running the System

Activate the environment and run:

streamlit run app_camera_infer_dual_freeze.py

The interface will open automatically in the browser.

---

# Sensor Wiring

E18-D80NK wiring to Arduino Uno:

| Sensor Wire | Arduino |
|------|------|
| Brown | 5V |
| Blue | GND |
| Black | Digital Pin 2 |

Serial communication:
Baudrate: 115200

---

# Features

✔ Low-cost industrial solution  
✔ CPU-only inference  
✔ Dual ROI inspection  
✔ Multiclass classification  
✔ Arduino-based automatic triggering  
✔ Operator-friendly interface  
✔ Production logging  
✔ Replicable installation procedure  

---

# Author

**André Gama de Matos**  
Engineer – Computer Vision & Embedded Systems  

Professional Master's Degree in Electrical Engineering  
Embedded Systems  

Universidade do Estado do Amazonas – UEA

Advisor  
Prof. Dr. Carlos Mauricio Seródio Figueiredo  

Co-advisor  
Prof. Dr. Jozias Parente de Oliveira  

---

# Academic Context

This system was developed as part of the Master's research project:

**"Computer Vision System for Automatic Inspection of Springs in Charger Covers"**

The project focuses on applying **deep learning and embedded systems** to industrial inspection scenarios.

---

# License

This project is intended for academic and industrial research purposes.
