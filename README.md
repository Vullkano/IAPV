<!--
Imitation Learning Suite - IAPV Part 3
======================================
-->

# 🧠 IAPV Suite - Imitation Learning

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/) 
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) 
[![Tech](https://img.shields.io/badge/tech-Streamlit%20|%20Gymnasium%20|%20SB3-purple)](https://streamlit.io/)

Bem-vindo ao **IAPV Suite**, um ambiente premium para treino, avaliação e visualização de agentes de **Aprendizagem por Imitação** (Behavioral Cloning e GAIL).
Este projeto foi desenhado para ser robusto, **bonito** e funcional, com suporte a **GridWorld** e **CartPole**.


## ⚡ Instalação Rápida

```bash
# 1. Instalar dependências
pip install -r requirements.txt
```


## 🖥️ Modo 1: Interface Web (Streamlit)

A experiência recomendada. Uma interface moderna ("Premium UI") que centraliza todo o fluxo de trabalho.

```bash
streamlit run src/streamlit.py
```

### ✨ Funcionalidades Premium
*   **Dashboard Intuitivo:** Visão geral do estado do sistema.
*   **Gravação Avançada:** Controlo D-Pad para GridWorld e geração automática via HuggingFace para CartPole.
*   **Terminal em Tempo Real:** Acompanhe o treino (train.py) diretamente no browser com logs limpos e organizados.
*   **Visualização:** Teste os modelos treinados com feedback visual imediato.


## ⌨️ Modo 2: Linha de Comandos (CLI)

Para automação e cumprimento estrito dos requisitos do projeto. Todos os scripts funcionam de forma independente.

### 1. Gravar Demonstrações (`demos.py`)
Recolhe dados de um especialista (humano ou AI).

```bash
# GridWorld (Manual - Use as Setas do Teclado)
python src/demos.py --gym Custom --episodes 10 --file output/demos_grid.pkl

# CartPole (Automático - Agente Pre-treinado HuggingFace)
python src/demos.py --gym CartPole --episodes 50 --file output/demos_cartpole.pkl --use-pretrained
```

### 2. Treinar Agente (`train.py`)
Treina um modelo (BC ou GAIL) usando os dados recolhidos.

```bash
# Treinar BC no GridWorld (100 Épocas)
python src/train.py --gym Custom --file output/demos_grid.pkl --output output/bc_grid.zip --algorithm BC --epochs 100

# Treinar GAIL no CartPole (200.000 Timesteps)
python src/train.py --gym CartPole --file output/demos_cartpole.pkl --output output/gail_cartpole.zip --algorithm GAIL --epochs 200
```

### 3. Executar & Visualizar (`run.py`)
Executa a política treinada no ambiente. Suporta modo interativo (passo-a-passo).

```bash
# Modo Contínuo (Execução Normal)
python src/run.py --file output/bc_grid.zip --gym Custom --algorithm BC --mode continuous

# Modo Passo-a-Passo (Pressione Enter para avançar)
python src/run.py --file output/bc_grid.zip --gym Custom --algorithm BC --mode step
```


## 📜 Descrição Detalhada dos Ficheiros

Para quem quer entender como o projeto funciona "por dentro", aqui está a explicação de cada componente:

### 🔹 Core (Obrigatórios)

| Ficheiro | Função |
| :--- | :--- |
| **`src/demos.py`** | **O "Gravador".** Responsável por recolher demonstrações. No GridWorld, captura os inputs das setas. No CartPole, carrega um agente especialista da HuggingFace para gerar dados perfeitos automaticamente. |
| **`src/train.py`** | **O "Cérebro".** Lê os dados gravados (`.pkl`) e treina um novo modelo. Suporta **BC** (Behavioral Cloning - Aprendizagem Supervisionada) e **GAIL** (Adversarial Learning). Guarda o resultado em `.zip`. |
| **`src/run.py`** | **O "Visualizador".** Carrega um modelo treinado (`.zip`) e coloca-o a interagir com o ambiente. Calcula métricas como taxa de sucesso e passos médios. Suporta execução passo-a-passo. |

### 🔹 Extras (Premium)

| Ficheiro | Função |
| :--- | :--- |
| **`src/streamlit.py`** | **O "Maestro".** Uma interface gráfica completa que une todos os scripts acima. Permite gravar, treinar e visualizar sem tocar no terminal. Inclui um dashboard e logs em tempo real. |
| **`src/custom_env.py`** | **O "Mundo".** A lógica do ambiente GridWorld. Define as regras do jogo: grelha 8x8, agente (2), objetivo (3), obstáculos (1) e recompensas. |

## 📁 Árvore do Projeto

```text
IAPV/
├── output/                   # 💾 Onde TUDO é guardado (Dados & Modelos)
│   ├── demos_grid.pkl        # Dados de demonstração
│   ├── bc_grid.zip           # Modelo treinado
│   └── ...
│
├── src/
│   ├── streamlit.py          # 🌐 Interface Principal
│   ├── demos.py              # 🎮 Script Gravação
│   ├── train.py              # 🧠 Script Treino
│   ├── run.py                # 🏃 Script Execução
│   ├── custom_env.py         # Lógica GridWorld
│   └── assets/               # 🎨 Estilos
│
└── requirements.txt          # Dependências do Python
```


## 👤 Autores
Trabalho realizado no âmbito da UC de **IAPV**.
- Diogo Alexandre Alonso de Freitas (Nº104841)
- João Francisco Marques Gonçalves da Silva Botas (Nº104782)
