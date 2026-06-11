# Q-Learning no Mundo de Wumpus 👾

Este repositório contém a implementação de um agente autônomo baseado em **Aprendizado por Reforço (Q-Learning)** para resolver o clássico problema do Mundo de Wumpus. O projeto foca em analisar os impactos da **transferência de aprendizado** entre ambientes com topologias diferentes.

## Sobre o Projeto

O Mundo de Wumpus é um tabuleiro 4x4 onde o agente precisa encontrar o ouro (presente em alguma casa do tabuleiro) e retornar ao ponto de partida em segurança. O ambiente contém poços e o monstro Wumpus. O agente desconhece o mapa e precisa aprender a política ótima apenas através de tentativa, erro e um sistema de recompensas.

**O sistema de recompensas estabelecido:**
* **+10** ao encontrar o ouro.
* **+10** ao retornar ao início com sucesso.
* **-10** (fatal) ao cair em um poço ou ser pego pelo Wumpus.
* **-2.0** e **-1.5** para passos adjacentes ao Wumpus e poços, respectivamente.
* **-0.5** por passo seguro, para evitar rotas longas ou loops. 

## Transferência de Aprendizado

O projeto visa analisar o comportamento do agente ao tentar reutilizar conhecimento prévio em um novo cenário.

1. **Mundo 1:** O agente foi treinado partindo de uma Q-table vazia e convergiu rapidamente (aprox. 570 episódios), descobrindo uma rota segura pela segunda linha do tabuleiro.
2. **Mundo 2:** O mapa foi alterado e a Q-table do Mundo 1 foi importada como base.

**Resultados:** Observa-se um caso clássico de **transferência negativa**. O viés geográfico do Mundo 1 atrapalhou a navegação no Mundo 2, exigindo **1.428 episódios** para a convergência. 

**Solução Aplicada:** Para superar o viés, os hiperparâmetros foram reajustados durante a transferência:
* Taxa de aprendizado aumentada de '0.15' para '0.50' para sobrescrever valores Q antigos rapidamente.
* Taxa de exploração resetada para '1.0' com um decay mais lento ('0.999'), forçando a re-exploração do novo mapa.

## Tecnologias Utilizadas
* **Python** (Lógica do ambiente e Q-Learning)
* **NumPy** (Operações de matriz e argmax)
* **Matplotlib** (Visualização das curvas de aprendizado)

## Como Executar

Clone o repositório e rode o script Python principal:

```bash
git clone [https://github.com/SeuUsuario/q-learning-wumpus.git](https://github.com/SeuUsuario/q-learning-wumpus.git)
cd q-learning-wumpus
python nome_do_seu_arquivo.py