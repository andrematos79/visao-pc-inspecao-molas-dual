# Inspeção de Molas — DUAL (v1.0.0 Stable)

Sistema de **inspeção visual automática** para verificação da presença de molas em carregadores de celular, utilizando **Visão Computacional + Deep Learning**, com interface em **Streamlit**, voltado para aplicação em **ambiente industrial**.

---

## 📌 Visão Geral

Este projeto implementa um sistema de inspeção **DUAL**, avaliando simultaneamente:
- Mola esquerda (ESQ)
- Mola direita (DIR)

Cada ROI é analisada por uma **CNN treinada** para classificação binária:
- `mola_presente`
- `mola_ausente`

O resultado final é:
- ✅ **APROVADO** → ambas as molas presentes
- ❌ **REPROVADO** → uma ou ambas ausentes

---

## 🧠 Arquitetura do Sistema

- **Frontend**: Streamlit (modo Operador / Engenharia)
- **Backend**: Python
- **Modelo**: TensorFlow / Keras
- **Aquisição**: Webcam USB (OpenCV)
- **Inferência**: ROI ESQ + ROI DIR
- **Configuração**: `config_molas.json`
- **Modelo treinado**: `modelo_molas.keras`

---

## 🖥️ Funcionalidades

- 📷 Captura via câmera USB
- 🖼️ Inferência via imagem carregada
- 🔍 ROIs independentes (ESQ / DIR)
- 📊 Contadores de Produção (Total, OK, NG, Yield)
- 🍩 Gráfico Donut de Yield
- 🔐 Modo Engenharia protegido por PIN
- ⚙️ Ajuste de ROI e Threshold via JSON
- 📈 Histórico e gráficos de qualidade

---

## 📂 Estrutura do Projeto (resumo)

