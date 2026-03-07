# 🧠 Inspeção de Molas — DUAL (Visão Computacional)

Sistema de **inspeção automática de molas metálicas em covers de carregadores**, 
baseado em **Visão Computacional + Deep Learning**, desenvolvido para aplicação 
em **linhas de produção industriais**, com foco em **automação, rastreabilidade e confiabilidade**.

O sistema integra **hardware de automação (Arduino + sensor de presença)** com 
**inferência por rede neural convolucional (CNN)** e uma interface de operação baseada em **Streamlit**.

---

# 📌 Visão Geral

O sistema realiza a inspeção simultânea de **duas molas metálicas (ESQ / DIR)** presentes em um cover de carregador.

Cada ROI é classificada por uma **CNN baseada em MobileNetV2** em três categorias:

- ✅ **OK** → mola presente e bem posicionada  
- ❌ **NG_MISSING** → mola ausente  
- ⚠️ **NG_MISALIGNED** → mola presente porém desalinhada  

A decisão final do produto utiliza uma **lógica DUAL conservadora**:

> O produto é aprovado **apenas se ambas as regiões forem classificadas como OK**.

---

# 🧠 Classificação Multiclasse

O modelo CNN possui **3 neurônios na camada de saída**:

| Classe | Descrição |
|------|------|
| OK | Mola presente e corretamente alinhada |
| NG_MISSING | Mola ausente |
| NG_MISALIGNED | Mola presente porém desalinhada |

Essa estratégia permite **diagnóstico mais preciso de defeitos**, importante para análise de processo e melhoria da produção.

---

# 🤖 Automação Industrial

O sistema foi projetado para operar de forma **automatizada em linha de produção**.

Fluxo operacional:

1️⃣ Operador posiciona o cover no dispositivo  
2️⃣ Sensor de presença detecta o produto  
3️⃣ Arduino Uno envia trigger via serial  
4️⃣ Sistema captura imagem automaticamente  
5️⃣ CNN realiza inferência nas duas ROIs  
6️⃣ Decisão DUAL é aplicada  
7️⃣ Resultado exibido na interface  
8️⃣ Dados registrados em log CSV / MES

---

# 🔌 Integração de Hardware

Hardware utilizado:

- Arduino Uno
- Sensor de proximidade (ex: E18-D80NK)
- Câmera USB / Industrial
- PC industrial ou workstation

O Arduino é responsável por:

- Detectar presença do produto
- Disparar a captura de imagem
- Sincronizar inspeção com a linha de produção

---

## 🧩 Arquitetura do Sistema

├── app_camera_infer_dual_freeze.py
├── models_registry.json
├── config_molas.json
├── configs/
├── models/
├── dataset_products/
├── logs/
├── assets/
├── docs/
└── requirements.txt

---

# 👷 Modos de Operação

## 👷 Operador

Interface simplificada para produção.

Funções disponíveis:

- Seleção do modelo
- Captura + inferência
- Visualização do resultado
- KPIs de produção
- Rastreamento de lote

Sem acesso a parâmetros críticos.

---

## 🛠️ Engenharia (PIN protegido)

Modo destinado a configuração do sistema.

Permite:

- Ajustar **ROI ESQ / DIR**
- Ajustar **threshold de decisão**
- Ativar normalização LAB
- Capturar imagens para dataset
- Criar **split train/val/test**
- Ajustar parâmetros por produto

🔐 PIN padrão: `1234`


---

# 🧠 Pipeline de Inferência

1️⃣ Captura do frame da câmera  
2️⃣ Recorte das ROIs (ESQ / DIR)  
3️⃣ Pré-processamento (opcional LAB normalization)  
4️⃣ Inferência CNN (MobileNetV2)  
5️⃣ Classificação multiclasse  
6️⃣ Aplicação da lógica **DUAL decision**  
7️⃣ Resultado final (OK / NG)  
8️⃣ Registro de dados no log

---

# 📊 Indicadores de Produção (KPIs)

A interface apresenta:

- Total inspecionado
- Quantidade OK
- Quantidade NG
- Yield (%)
- Histórico de inspeções
- Defeitos por lado (ESQ / DIR)

---

# 🧾 Rastreabilidade

O sistema permite rastrear cada unidade produzida.

Campos registrados:

- Timestamp
- Modelo do produto
- Resultado da inspeção
- Classe inferida
- Probabilidades da CNN
- Serial Number (opcional)
- Ordem de Produção
- Operador

Logs são armazenados em: logs/YYYY-MM-DD.csv

---

# 🏭 Integração MES (opcional)

O sistema possui suporte para integração com **MES (Manufacturing Execution System)**.

Quando habilitado:

- Resultados podem ser enviados ao MES
- Serial Number pode ser associado ao produto
- Dados de produção podem ser sincronizados

O MES pode ser ativado/desativado via interface.


# 📦 Dataset de Treinamento

O dataset utilizado neste projeto suporta **classificação multiclasse** para inspeção automática das molas metálicas presentes no cover do carregador.

## Classes de inspeção

- `OK` → mola presente e corretamente montada
- `NG_MISSING` (`ng_ausente`) → mola ausente
- `NG_MISALIGNED` (`ng_desalinhada`) → mola presente, porém desalinhada

## Estrutura atual

```text
dataset/
├── ok/
├── ng_ausente/
├── ng_desalinhada/
├── split2/
│   ├── train/
│   │   ├── ok/
│   │   ├── ng_ausente/
│   │   └── ng_desalinhada/
│   ├── val/
│   │   ├── ok/
│   │   ├── ng_ausente/
│   │   └── ng_desalinhada/
│   └── test/
│       ├── ok/
│       ├── ng_ausente/
│       └── ng_desalinhada/
└── split_aug/
    ├── train/
    │   ├── ok/
    │   ├── ng_ausente/
    │   └── ng_desalinhada/
    ├── val/
    │   ├── ok/
    │   ├── ng_ausente/
    │   └── ng_desalinhada/
    └── test/
        ├── ok/
        ├── ng_ausente/
        └── ng_desalinhada/

Observação importante

A organização do dataset foi projetada para suportar treinamento e avaliação de modelos CNN com classificação multiclasse, alinhada à versão mais recente do sistema e ao artigo científico associado ao projeto.

Finalidade

Essa estrutura permite treinar, validar e testar modelos CNN capazes de distinguir entre:

condições normais de montagem

ausência de mola

desalinhamento da mola

Essa abordagem é fundamental para aumentar a capacidade diagnóstica do sistema em ambiente industrial.


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

