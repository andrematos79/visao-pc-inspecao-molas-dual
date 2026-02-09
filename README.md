# 🧠 Inspeção de Molas — DUAL (Visão Computacional)

Sistema de **inspeção automática de molas** baseado em **Visão Computacional + Deep Learning**, desenvolvido para aplicação em **linha de produção**, com foco em **estabilidade, rastreabilidade e separação clara entre Operador e Engenharia**.

---

## 📌 Visão Geral

O sistema realiza a inspeção simultânea de **duas molas (ESQ / DIR)** presentes em um cover de carregador, classificando cada amostra como **APROVADA (OK)** ou **REPROVADA (NG)** com base na probabilidade inferida por um modelo CNN treinado.

Principais características:
- 🔍 Inspeção DUAL (ESQ + DIR)
- 🧠 Modelo CNN em TensorFlow/Keras
- 🎥 Captura via câmera USB / industrial
- 🧑‍🏭 Modo Operador (produção)
- 🛠️ Modo Engenharia (setup protegido por PIN)
- 📊 KPIs de produção (Total, OK, NG, Yield)
- 🗂️ Dataset estruturado automaticamente por produto
- 🔄 Configuração **independente por modelo**

---

## 🧩 Arquitetura do Sistema

├── app_camera_infer_dual_freeze.py # App principal (Streamlit)
├── models_registry.json # Cadastro de modelos/linhas
├── config_molas.json # Configuração default (fallback)
├── configs/ # Config por modelo (auto-gerado)
├── labels.json # Classes do modelo
├── assets/ # Logos e recursos visuais
├── logs/ # Logs CSV por data
├── dataset_products/ # Dataset de aprendizado (auto)
└── requirements.txt


---

## 👷‍♂️ Modos de Operação

### 👷 Operador
- Apenas **seleção do modelo**
- Captura + inferência
- Visualização do resultado
- KPIs de produção
- ❌ Sem acesso a ROI, threshold ou configs

### 🛠️ Engenharia (PIN protegido)
- Ajuste de **ROI ESQ / DIR**
- Ajuste de **threshold**
- Normalização LAB
- Salvamento de config por modelo
- Captura de imagens para **dataset**
- Geração de **split train/val/test**

> 🔐 PIN padrão: `1234` (alterar em produção)

---

## 🧠 Pipeline de Inferência

1. Captura de frame da câmera
2. Recorte das ROIs (%)
3. (Opcional) Normalização LAB
4. Inferência CNN
5. Cálculo da probabilidade `mola_presente`
6. Decisão por threshold
7. Resultado final (OK / NG)
8. Log CSV + atualização de KPIs

---

## 📊 Indicadores (KPIs)

- Total inspecionado
- OK / NG
- Yield (%)
- Tempo de teste (s)
- Histórico acumulado
- Gráficos de tendência (Yield e defeitos por lado)

---

## 📁 Dataset de Aprendizado

Estrutura automática por produto:

dataset_products/
└── PRODUTO_X/
├── raw/
│ ├── ok/
│ └── ng/
├── roi/
│ ├── ESQ/
│ │ ├── mola_presente/
│ │ └── mola_ausente/
│ └── DIR/
│ ├── mola_presente/
│ └── mola_ausente/
└── roi_split/
├── ESQ/
│ ├── train/
│ ├── val/
│ └── test/
└── DIR/
├── train/
├── val/
└── test/


---

## ⚙️ Requisitos

- Python 3.10+
- OpenCV
- TensorFlow
- Streamlit
- NumPy
- Matplotlib (opcional para gráficos)

Instalação:
```bash
pip install -r requirements.txt

Execução:

streamlit run app_camera_infer_dual_freeze.py

🏷️ Versionamento

v1.0.0 → Baseline estável de produção (tagged)

Branch main → produção

Branch develop → evolução

🏭 Aplicação Industrial

Este sistema foi projetado para:

Operar em linha de produção real

Evitar ajustes acidentais por operadores

Garantir rastreabilidade

Permitir rápida troca de produto/modelo

Servir como base para evolução (v1.1.0+)

## 🎓 Contexto Acadêmico

Este software foi desenvolvido no âmbito do **Curso de Mestrado em Engenharia Elétrica**, 
com ênfase em **Sistemas Embarcados**, da **Universidade do Estado do Amazonas (UEA)**.

O desenvolvimento deste sistema integra as atividades de pesquisa aplicada do trabalho de mestrado, 
sob a orientação do **Professor Doutor Carlos Mauricíco Seródio Figueiredo**, 
com foco em soluções de **Visão Computacional aplicada à Automação Industrial**, alinhadas aos conceitos da **Indústria 4.0**.

Os resultados obtidos contribuem para a investigação de técnicas de inspeção visual automatizada em ambientes industriais, 
bem como para a validação prática de arquiteturas baseadas em **Deep Learning** e **Sistemas Embarcados** em linhas de produção reais.


👨‍💻 Autor

André Gama de Matos
Engenheiro de Software / Software Engineer
Visão Computacional • Sistemas Embarcados • Indústria 4.0

📌 Licença

Uso interno / educacional / industrial conforme política do projeto.

> Este projeto possui finalidade acadêmica e de pesquisa aplicada, podendo ser utilizado como base 
> para estudos, desde que devidamente referenciado.

