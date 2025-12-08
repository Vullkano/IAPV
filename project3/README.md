# Projeto de Aprendizagem por Imitação (Imitation Learning)

Este repositório contém a implementação de algoritmos de **Imitation Learning** (Aprendizagem por Imitação) aplicados a dois ambientes:
1.  **CartPole-v1** (Clássico problema de controlo).
2.  **GridWorld-v0** (Ambiente de navegação customizado com obstáculos aleatórios).

O objetivo é treinar agentes que aprendem a resolver estes problemas não através de recompensas (Reinforcement Learning), mas sim **observando demonstrações de um perito** (humano ou sintético).

---

## 📚 Teoria: O que estamos a fazer?

### 1. Behavior Cloning (BC)
O **Behavior Cloning** é a forma mais simples de imitação. Funciona como **Aprendizagem Supervisionada** clássica.
*   **Ideia:** "Se o perito estava no estado X e fez a ação Y, eu também devo fazer Y."
*   **Vantagens:** Simples de implementar e rápido de treinar.
*   **Desvantagens:** Sofre de "Desvio de Distribuição" (Covariate Shift). Se o agente cometer um pequeno erro e for parar a um estado que nunca viu nas demonstrações, não sabe como recuperar e o erro acumula-se.

### 2. Generative Adversarial Imitation Learning (GAIL)
O **GAIL** é inspirado nas GANs (Generative Adversarial Networks).
*   **Ideia:** Temos duas redes a competir:
    *   **Gerador (O Agente):** Tenta agir de forma tão parecida com o perito que confunde o discriminador.
    *   **Discriminador:** Tenta distinguir entre uma trajetória feita pelo perito e uma feita pelo agente.
*   **Vantagens:** Aprende uma política mais robusta a longo prazo, pois o agente é treinado para corrigir os seus erros (através do feedback do discriminador).
*   **Desvantagens:** Mais difícil de treinar (instabilidade) e computationally expensive.

---

## 🛠️ Detalhes Técnicos

### Estrutura do Projeto
*   `demos.py`: Script para **recolha de dados**. Permite jogar manualmente (Custom) ou usar um modelo pré-treinado (CartPole).
*   `train.py`: Script de **treino**. Implementa BC e GAIL usando a biblioteca `imitation`.
*   `run.py`: Script de **visualização**. Carrega os modelos treinados e mostra-os em ação.
*   `custom_env.py`: Implementação do ambiente **GridWorld** compatível com Gymnasium.

### O Ambiente Customizado (`GridWorld-v0`)
Um labirinto $8 \times 8$ gerado aleatoriamente a cada episódio.
*   **Estado (Observação):** O agente recebe um vetor de 8 números inteiros:
    *   `[Agente_X, Agente_Y]` (Posição Atual)
    *   `[Parede_Cima, Parede_Baixo, Parede_Esq, Parede_Dir]` (Sensores de proximidade: 1=Parede, 0=Livre)
    *   `[Distancia_X_Meta, Distancia_Y_Meta]` (Posição Relativa do Objetivo)
*   **Ações:** 4 Movimentos (0=Direita, 1=Cima, 2=Esquerda, 3=Baixo).
*   **Recompensa:** (Usada apenas para avaliação, não para o treino de imitação) +1 ao chegar à meta, -0.05 por passo.

---

## 🚀 Guia Prático Passo-a-Passo

### 0. Instalação
Certifica-te que tens as dependências instaladas:
```bash
pip install -r requirements.txt
```

### 1. Recolha de Demonstrações (`demos.py`)
Antes de treinar, precisamos de dados ("exemplos") para o agente imitar.

**A. Para o GridWorld (Tu és o Perito!)**
Vais controlar o agente com as **Setas do Teclado**. Tenta ser o mais eficiente possível (caminho mais curto). O "Professor" aqui é um **Humano**.
```bash
python demos.py --gym Custom --episodes 10 --file output/demos_grid.pkl
```

**B. Para o CartPole (Perito Sintético/Artificial)**
Equilibrar um pau com o teclado é difícil e aborrecido. Por isso, usamos um **"Perito Sintético"**.
*   **O que é?** É uma inteligência artificial (PPO) que *já foi treinada antes* por outra pessoa e colocada na internet (HuggingFace).
*   **Para que serve?** Nós fazemos download desse "cérebro" já ensinado e pedimos-lhe para jogar CartPole enquanto gravamos. Assim temos demonstrações perfeitas sem termos trabalho manual.
```bash
python demos.py --gym CartPole --episodes 50 --file output/demos_cartpole.pkl --use-pretrained
```

### 2. Treino (`train.py`)
Agora vamos ensinar o agente. Podes escolher o algoritmo (`BC` ou `GAIL`) e o ambiente.

**Treino com Behavior Cloning (BC)**
```bash
# Treinar no GridWorld
python train.py --gym Custom --file output/demos_grid.pkl --output output/bc_grid.zip --algorithm BC

# Treinar no CartPole
python train.py --gym CartPole --file output/demos_cartpole.pkl --output output/bc_cartpole.zip --algorithm BC
```

**Treino com GAIL**
```bash
# Treinar no GridWorld
python train.py --gym Custom --file output/demos_grid.pkl --output output/gail_grid.zip --algorithm GAIL

# Treinar no CartPole
python train.py --gym CartPole --file output/demos_cartpole.pkl --output output/gail_cartpole.zip --algorithm GAIL
```

### 3. Execução e Teste (`run.py`)
Vê o resultado final! O script tem um modo "Passo-a-passo" (ideal para o GridWorld) e "Contínuo" (ideal para CartPole).

```bash
## GridWorld ##
# Ver o Behavior Cloning (BC) - GridWorld
python run.py --gym Custom --file output/bc_grid.zip

# Ver o Generative Adversarial Imitation Learning (GAIL) - GridWorld
python run.py --gym Custom --file output/gail_grid.zip

## CartPole ##
# Ver o Behavior Cloning (BC) - CartPole
python run.py --gym CartPole --file output/bc_cartpole.zip

# Ver o Generative Adversarial Imitation Learning (GAIL) - CartPole
python run.py --gym CartPole --file output/gail_cartpole.zip
```

---

## 📊 O que esperar dos Resultados?

1.  **GridWorld:**
    *   O **BC** deve conseguir resolver a maioria dos mapas se as demonstrações cobrirem bem as situações (ex: contornar paredes). Se tiveres poucos dados, ele pode ficar preso em cantos("Corner trap").
    *   O **GAIL** pode demorar mais a convergir, mas tende a encontrar caminhos mais robustos.

2.  **CartPole:**
    *   O **BC** resolve este problema facilmente (o "pau" não deve cair).
    *   É o teste perfeito para validar se o pipeline está a funcionar.
