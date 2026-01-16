# 🧠 IAPV Suite - Imitation Learning

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/) 
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) 
[![Tech](https://img.shields.io/badge/tech-Streamlit%20|%20Gymnasium%20|%20SB3-purple)](https://streamlit.io/)

Bem-vindo ao **IAPV Suite**, um ambiente premium para treino, avaliação e visualização de agentes de **Aprendizagem por Imitação** (Behavioral Cloning e GAIL).
Este projeto foi desenhado para ser robusto, **bonito** e funcional, com suporte a **GridWorld** e **CartPole**.

---

## ⚡ Instalação Rápida

```bash
# 1. Instalar dependências (Gymnasium, SB3, Streamlit, etc)
pip install -r requirements.txt
```

---

## 🖥️ Modo 1: Interface Web (Streamlit - "AI LAB")

A experiência central do projeto. Uma interface moderna com estética *Dark Mode* e *Glassmorphism* que centraliza todo o fluxo de trabalho de um investigador de IA.

```bash
streamlit run src/streamlit.py
```

### ✨ Funcionalidades Premium

*   **Mission Control UI:** Painéis de controlo estilizados para gravação e treino, com feedback visual em tempo real.
*   **Modo de Spawn (GridWorld):** Alternância entre spawn **Aleatório** (🎲) e **Fixo** (📍) para testar a capacidade de generalização.
*   **Gravação Híbrida:** 
    *   **GridWorld:** Controlo via D-Pad (Web) ou CLI (Teclado).
    *   **CartPole:** Geração automática via HuggingFace ou modo manual experimental com mapping de setas.
*   **Laboratório de Treino:** Configuração dinâmica de parâmetros (Epochs, Seeds, Validação) com logs estilo terminal integrados.
*   **Robustez Estatística:** Sistema automático de validação que utiliza janelas de sementes dinâmicas (`seed + i`) para gerar distribuições de recompensa reais no box plot.
*   **Benchmark Automatizado:** Treino em batch com datasets incrementais para analisar curvas de aprendizagem.

---

## 📊 Análise de Robustez e Optimização

O IAPV Suite implementa várias melhorias para garantir resultados académicos e tempos de execução rápidos:

1.  **Epochs Diferenciadas:** 
    *   **CartPole:** Otimizado para **10 Epochs** (Suficiente para convergência rápida devido à qualidade dos dados).
    *   **GridWorld:** Padrão de **100 Epochs** para garantir a resolução de labirintos complexos.
2.  **Validação Stochastic:** O sistema de benchmark corre agora **20 simulações** por modelo. Cada simulação usa uma semente diferente derivada da semente base (`base_seed + 1000 + i`), garantindo que o **Box Plot de Robustez** mostre uma variabilidade real do ambiente e não apenas a repetição do mesmo cenário.

---

## ⌨️ Modo 2: Linha de Comandos (CLI)

Todos os scripts são utilizáveis de forma independente para integração em pipelines ou scripts de shell.

### 1. Gravar Demonstrações (`demos.py`)
Recolhe trajetórias de um especialista.

```bash
# GridWorld: Spawn Aleatório (Use as SETAS do teclado)
python src/demos.py --gym Custom --episodes 10 --file output/grid/random/simple/demos.pkl --spawn random

# CartPole: Gerar via Agente HuggingFace (Totalmente automático)
python src/demos.py --gym CartPole --episodes 50 --file output/cartpole/demos.pkl --use-pretrained

# CartPole: Treinar MANUALMENTE (Para diversão/teste)
python src/demos.py --gym CartPole --episodes 3 --file output/cartpole/manual_test.pkl
```

### 2. Treinar Agente (`train.py`)
Lê as demonstrações (`.pkl`) e gera o modelo neuronal (`.zip`).

```bash
# Treinar BC no GridWorld (Spawn Aleatório) com Seed específica
python src/train.py --gym Custom --file output/grid/random/simple/demos.pkl --output output/grid/random/simple/bc_grid.zip --algorithm BC --epochs 100 --spawn random --seed 42

# Treinar GAIL no CartPole (Rápido: 10 épocas)
python src/train.py --gym CartPole --file output/cartpole/demos.pkl --output output/cartpole/gail_cartpole.zip --algorithm GAIL --epochs 10
```

### 3. Visualizar o Resultado (`run.py`)
Executa a política aprendida e mostra o agente em ação.

```bash
# GridWorld: Modo Passo-a-Passo (Pressione ENTER para cada passo)
python src/run.py --file output/grid/random/simple/bc_grid.zip --gym Custom --algorithm BC --mode step --spawn random

# CartPole: Visualização Contínua
python src/run.py --file output/cartpole/gail_cartpole.zip --gym CartPole --algorithm GAIL --mode continuous
```

---

## 📁 Estrutura de Diretórios Standard

O Suite organiza automaticamente os ficheiros para evitar conflitos de dados:

```text
IAPV/
├── output/
│   ├── grid/                    # 🧭 Ambiente GridWorld
│   │   ├── random/              # Spawn Aleatório (Pasta Principal)
│   │   │   ├── simple/              # Testes Únicos
│   │   │   └── intervals/           # Benchmark Data & Plots
│   │   └── fixed/               # Spawn Fixo (📍)
│   └── cartpole/                # 🕹️ Ambiente CartPole
│       ├── demos.pkl            # Dataset Principal
│       ├── manual_test.pkl      # Testes Manuais
│       └── intervals/           # Modelos de Benchmark (1, 2, 3... demos)
│
├── docs/                        # 🖼️ Documentação e Gráficos
│   └── plots/                   # Gráficos de Sucesso, Reward e Robustez
│
├── src/                         # 🛠️ Código Fonte
│   ├── streamlit.py             # Interface (O Maestro)
│   ├── demos.py                 # Gravador
│   ├── train.py                 # Treinador
│   ├── run.py                   # Visualizador
│   ├── custom_env.py            # Lógica Gymnasium
│   └── extras/                  # Motor de Benchmark e Plots
```

---

## 📜 Glossário de Ficheiros

*   **`.pkl` (Demos):** É a "cassete" com os dados brutos. Se abrires, verás as observações e ações do especialista. É a matéria-prima para o treino.
*   **`.zip` (Modelos):** É o "cérebro" treinado. Contém os pesos da rede neuronal que o agente usa para decidir que ação tomar em cada estado.

---

## 🚀 Notas de Execução

- **Teclado (Manual):** Nos modos manuais (`demos.py`), utilize as **Setas** para movimento. Pressione **ESC** para sair antecipadamente e salvar os dados recolhidos até ao momento.
- **Gráficos:** No tab "Relatório", utilize o botão **"Atualizar Gráficos"** após o treino em batch para regenerar as imagens em `docs/plots/`.
- **Compatibilidade:** O código utiliza as versões mais recentes do **Gymnasium** e **Stable Baselines 3**, garantindo longevidade e suporte a novas funcionalidades de RL.

---

## 👤 Autores

Trabalho desenvolvido para a Unidade Curricular de **IAPV**.
- Diogo Alexandre Alonso de Freitas (Nº104841)
- João Francisco Marques Gonçalves da Silva Botas (Nº104782)
- Miguel Gonçalves Pereira (Nº105944)
