*********/
# SVC – Sistema de Visão Computacional para Inspeção Automatizada de Molas em Carregadores

## Visão Geral

Este projeto apresenta um **sistema de visão computacional de baixo custo para inspeção automática de molas metálicas** instaladas dentro de carregadores de celular.

O sistema utiliza:

- **Câmera USB**
- **Rede neural convolucional (MobileNetV2)**
- **Lógica de decisão DUAL baseada em duas regiões de interesse (ROI)**

O objetivo é classificar automaticamente a **presença e o alinhamento das molas** no interior do produto.

A solução foi desenvolvida no contexto do **Mestrado Profissional em Engenharia Elétrica – Sistemas Embarcados** da **Universidade do Estado do Amazonas (UEA)**.

O sistema opera em **ambiente industrial real utilizando apenas CPU**, integrando hardware e software para inspeção automatizada em linha de produção.

Este repositório representa a **versão final utilizada no chão de fábrica**, evoluindo de um **protótipo de laboratório para um sistema industrial validado em produção**.

---

# Contexto Industrial

Na produção de carregadores de celular, as molas internas são responsáveis pelo **contato elétrico adequado com os terminais do conector**.

Defeitos nessas molas podem causar:

- mau contato
- falha de funcionamento
- retrabalho ou descarte do produto

Tradicionalmente essa inspeção é realizada manualmente por operadores, o que pode introduzir:

- variabilidade humana
- fadiga visual
- inconsistência no critério de decisão

O **SVC (Sistema de Visão Computacional)** substitui essa inspeção manual por um **processo automatizado baseado em inteligência artificial**, aumentando a confiabilidade e repetibilidade da inspeção.

---

# Evolução do Projeto

O sistema passou por duas fases principais de desenvolvimento.

## Fase 1 – Protótipo de Laboratório

Nesta fase foram realizados:

- criação inicial do dataset
- experimentos com redes neurais convolucionais
- validação do conceito de inspeção
- desenvolvimento da lógica de decisão DUAL

## Fase 2 – Sistema Industrial

Após validação inicial, o sistema foi adaptado para condições reais de produção:

- variações de iluminação
- posicionamento da câmera
- fluxo contínuo da linha de produção
- melhorias na interface do operador
- ajustes de robustez para ambiente fabril

A versão presente neste repositório corresponde à **versão estabilizada e validada na linha de produção**.

Essa evolução representa um aumento significativo no **Technology Readiness Level (TRL)** do sistema.

---

# Arquitetura do Sistema

O fluxo de operação do sistema é composto pelas seguintes etapas:

Câmera USB  
↓  
Captura da imagem  
↓  
Extração das ROIs (Esquerda / Direita)  
↓  
Classificação com CNN MobileNetV2  
↓  
Lógica de decisão DUAL  
↓  
Visualização do resultado + registro de produção

O sistema também possui **disparo automático por sensor de proximidade conectado a um Arduino Uno**.


![Arquitetura do SVC](docs/figures/svc_architecture_diagram.png)


---

# Componentes de Hardware

| Componente | Descrição |
|--------|--------|
| Computador | Windows 11 Pro |
| CPU | Intel Core i3 12ª geração ou superior |
| RAM | 8 GB ou mais |
| Câmera | Câmera USB industrial |
| Microcontrolador | Arduino Uno |
| Sensor | Sensor de proximidade E18-D80NK |
| Interface | USB |

O sensor dispara automaticamente a inspeção quando o produto é posicionado diante da câmera.

---

# Stack de Software

O sistema foi desenvolvido em Python utilizando bibliotecas modernas de visão computacional e aprendizado de máquina.

| Software | Função |
|--------|--------|
| Python | Linguagem principal |
| TensorFlow / Keras | Inferência da rede neural |
| OpenCV | Processamento de imagens |
| Streamlit | Interface do operador |
| PySerial | Comunicação com Arduino |
| Pandas | Registro de produção |
| Matplotlib | Visualização |
| Streamlit Autorefresh | Monitoramento automático do sensor |

---

# Modelo de Inteligência Artificial

O sistema utiliza **MobileNetV2 com Transfer Learning** para classificar o estado das molas.

## Classes do Modelo

| Classe | Descrição |
|------|------|
| OK | Mola corretamente instalada |
| NG_MISSING | Mola ausente |
| NG_MISALIGNED | Mola presente porém desalinhada |

Cada lado do produto é avaliado independentemente.

---

# Lógica de Decisão DUAL

A decisão final segue uma regra industrial conservadora.

O produto é aprovado somente se:

ROI esquerda = OK  
E  
ROI direita = OK

Se qualquer lado for classificado como defeito, o produto é rejeitado.

Essa abordagem aumenta significativamente a confiabilidade da inspeção.

---

# Modos de Operação

## Modo Manual

O operador dispara a inspeção manualmente.

## Modo Automático

A inspeção é disparada automaticamente pelo sensor **E18-D80NK conectado ao Arduino**.

Mensagens enviadas via serial:

PRESENT=0  
PRESENT=1

A interface Streamlit monitora a porta serial e executa a inferência.

---

# Recursos Industriais Avançados

## Coleta Automática de Dataset

Durante o desenvolvimento inicial, a criação do dataset exigia **extração manual de prints da tela**, processo extremamente demorado.

Foi então implementado um **módulo automático de captura de dataset**.

Esse recurso permite:

- captura automática de imagens da câmera
- salvamento automático das ROIs
- geração de dataset durante a operação do sistema

Isso reduziu o tempo de criação de datasets de **vários dias para poucas horas**.

---

## Recurso “Aprovado com Atenção”

O sistema detecta situações onde a classificação está **muito próxima do limite de decisão da rede neural**.

Exemplo:

Probabilidade = 0.51  
Threshold = 0.50

Nesse caso:

- o produto é aprovado
- o sistema sinaliza uma **aprovação no limite**

Esse recurso ajuda a identificar **tendências de degradação do processo produtivo**.

---

## Registro de Evidências para Auditoria

O sistema possui um módulo de **registro automático de evidências**.

Imagens podem ser armazenadas quando:

- defeitos são detectados
- aprovações próximas do limite ocorrem
- engenharia precisa investigar o processo

As imagens são armazenadas em:

C:\SVC_INSPECAO_MOLAS\dataset_auto_evidencias

Isso permite:

- auditoria de qualidade
- análise de defeitos
- expansão do dataset

---

## Monitoramento de Uso de Disco

Como o sistema pode salvar muitas imagens, foi implementado um **monitoramento automático do uso de disco**.

O operador pode configurar um limite máximo de armazenamento.

Quando esse limite é atingido, o sistema gera um alerta.

---

## Política de Retenção com Auto Delete

O sistema permite **exclusão automática de evidências antigas**.

O tempo de retenção pode ser configurado para:

- 30 dias
- 60 dias
- 90 dias

Isso evita saturação de disco e garante operação contínua.

---
---

# Resultados de Validação Industrial

Após o desenvolvimento e treinamento do modelo, o sistema foi avaliado em condições representativas do ambiente industrial.

## Dataset

Para treinamento e validação do modelo foi utilizado um **dataset proprietário composto por aproximadamente 1170 imagens reais** capturadas diretamente da linha de produção.

As imagens foram coletadas utilizando a mesma câmera utilizada no sistema final, garantindo consistência entre o ambiente de treinamento e operação.

As imagens foram organizadas em três classes:

| Classe | Descrição |
|------|------|
| OK | Mola corretamente posicionada |
| NG_MISSING | Mola ausente |
| NG_MISALIGNED | Mola presente porém desalinhada |

Cada imagem contém duas regiões de interesse correspondentes às posições esquerda e direita da mola.
Os testes experimentais foram conduzidos com 100 peças reais, distribuídas em 50 amostras OK, 30 NG_MISALIGNED e 20 NG_MISSING. Os resultados reforçam a aplicabilidade do sistema em ambiente industrial e sua evolução de protótipo laboratorial para solução validada em produção.
![Validação industrial do SVC](docs/figures/svc_validation_results.png)
---

## Teste Experimental em Peças Reais

Após o treinamento, o sistema foi avaliado utilizando **100 unidades reais de carregadores**.

Distribuição do conjunto de teste:

| Classe | Quantidade |
|------|------|
| OK | 50 |
| NG_MISALIGNED | 30 |
| NG_MISSING | 20 |

Durante o teste, o sistema executou:

- captura automática de imagem
- recorte das regiões de interesse
- inferência da rede neural
- aplicação da lógica DUAL
- registro do resultado

---

## Desempenho do Sistema

Durante os testes experimentais foram observados os seguintes resultados:

- **Alta acurácia de classificação**
- detecção consistente de desalinhamentos
- identificação confiável de ausência de mola

A lógica **DUAL baseada em duas ROIs** mostrou-se particularmente eficaz para reduzir falsos positivos e aumentar a robustez do sistema.

---

## Tempo de Ciclo

O tempo médio de processamento observado foi aproximadamente:
1.93 segundos por unidade


Esse tempo inclui:

- captura da imagem
- processamento da rede neural
- tomada de decisão
- registro do resultado

Esse desempenho é compatível com o fluxo de produção da linha.

---

## Robustez em Ambiente Industrial

Durante a operação na linha de produção, o sistema demonstrou robustez frente a:

- variações de iluminação
- pequenas variações de posicionamento do produto
- ruído visual do ambiente industrial

Os recursos adicionais implementados, como:

- coleta automática de dataset  
- registro de evidências  
- monitoramento de disco  
- política de retenção automática  

contribuem para a **manutenção e evolução contínua do sistema**.

---

## Impacto Industrial

A substituição da inspeção manual por inspeção automatizada proporciona:

- redução de variabilidade humana
- aumento da repetibilidade da inspeção
- maior rastreabilidade do processo
- padronização do critério de decisão

Esses fatores tornam o sistema adequado para aplicações industriais de inspeção visual automatizada.

---

# Estrutura do Projeto

SVC_INSPECAO_MOLAS
│
├── app_camera_infer_dual_freeze.py
├── config_molas.json
├── labels.json
├── labels_L2.json
├── models_registry.json
├── requirements.txt
│
├── assets
│ └── logo_empresa.jpg
│
├── configs
│ └── UNICORN_WHITE_15W.json

---

# Instalação

Criar pasta do projeto:

C:\SVC_INSPECAO_MOLAS

Criar ambiente virtual:

python -m venv .venv_svc

Ativar ambiente:

.\.venv_svc\Scripts\Activate.ps1

Instalar dependências:

pip install -r requirements.txt

---

# Execução do Sistema

streamlit run app_camera_infer_dual_freeze.py

A interface será aberta automaticamente no navegador.

---

# Conexão do Sensor

| Fio do Sensor | Arduino |
|------|------|
| Marrom | 5V |
| Azul | GND |
| Preto | Pino Digital 2 |

Comunicação serial: **115200 baud**

---

# Principais Características

✔ Solução industrial de baixo custo  
✔ Inferência apenas em CPU  
✔ Inspeção com duas ROIs  
✔ Classificação multiclasse  
✔ Disparo automático via Arduino  
✔ Coleta automática de dataset  
✔ Registro de evidências  
✔ Monitoramento de uso de disco  
✔ Política automática de retenção  
✔ Detecção de aprovação no limite  
✔ Registro de produção  
✔ Sistema validado em ambiente industrial  

---

# Autor

**André Gama de Matos**

Engenheiro – Visão Computacional e Sistemas Embarcados  

Mestrado Profissional em Engenharia Elétrica – Sistemas Embarcados  
Universidade do Estado do Amazonas – UEA

Orientador:  
Prof. Dr. Carlos Maurício Seródio Figueiredo  

Coorientador:  
Prof. Dr. Jozias Parente de Oliveira

---

# Contexto Acadêmico

Este sistema foi desenvolvido no contexto da dissertação:

**Sistema de Visão Computacional para Inspeção Automatizada de Molas em Carregadores de Celular**

O trabalho investiga a aplicação de **deep learning e sistemas embarcados para inspeção industrial automatizada**.

---

# Licença

Projeto destinado a fins **acadêmicos e de pesquisa industrial**.
