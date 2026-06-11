
import numpy as np
import random
import json 

# Aplicação de aprendizado por reforço no clássico jogo no Mundo de Wumpus.

class MundoWumpus:
    """
    Tabuleiro 4x4 com coordenadas (linha, coluna), variando de 1 a 4.
    A posição (1, 1) é o canto inferior esquerdo: ponto de início e retorno.
    """

    def __init__(self, pocos, wumpus, ouro):
        self.tamanho = 4
        self.pocos   = pocos
        self.wumpus  = wumpus
        self.ouro    = ouro
        self.inicio  = (1, 1)
        self.reiniciar()

    def reiniciar(self):
        self.pos_agente = self.inicio
        self.tem_ouro   = False # o estado deve guardar se o ouro foi obtido ou não para saber se pode retornar ao início.
        return self._estado()

    def _estado(self):
        """Estado = (linha, coluna, tem_ouro)."""
        return (self.pos_agente[0], self.pos_agente[1], int(self.tem_ouro))

    def _adjacente(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    def exibir_mapa(self):
        print("+" + "---+" * self.tamanho)
        for linha in range(self.tamanho, 0, -1):
            linha_str = "|"
            for col in range(1, self.tamanho + 1):
                pos = (linha, col)
                if   pos == self.wumpus:   celula = " W "
                elif pos in self.pocos:    celula = " P "
                elif pos == self.ouro:     celula = " O "
                elif pos == self.inicio:   celula = " I "
                else:                      celula = "   "
                linha_str += celula + "|"
            print(linha_str)
            print("+" + "---+" * self.tamanho)
        print("O que as letras representam?: [I] Início [O] Ouro  [W] Wumpus  [P] Poço\n")

    def passo(self, acao):
        """
        Ações: 0=Cima, 1=Baixo, 2=Esquerda, 3=Direita.
        Retorna: (estado, recompensa, encerrado, vitoria)
        """
        linha, col = self.pos_agente
        if   acao == 0 and linha < self.tamanho: linha += 1
        elif acao == 1 and linha > 1:            linha -= 1
        elif acao == 2 and col   > 1:            col   -= 1
        elif acao == 3 and col   < self.tamanho: col   += 1

        self.pos_agente = (linha, col)
        estado = self._estado()

        if self.pos_agente in self.pocos:  return estado, -10, True, False
        if self.pos_agente == self.wumpus: return estado, -10, True, False

        recompensa = 0

        if self.pos_agente == self.ouro and not self.tem_ouro:
            self.tem_ouro = True
            recompensa   += 10
            estado        = self._estado()

        if self.pos_agente == self.inicio and self.tem_ouro:
            return estado, recompensa + 10, True, True

        # Penaliza com -0.5 toda vez que avançar para uma nova casa para evitar que o algoritmo fique andando em círculos.
        penalidade = -0.5
        if self._adjacente(self.pos_agente, self.wumpus):
            penalidade = -2.0
        for poco in self.pocos:
            if self._adjacente(self.pos_agente, poco):
                penalidade = max(penalidade, -1.5)

        return estado, recompensa + penalidade, False, False


# Aplicação do algoritmo Q-learning

def treinar_q_learning(
    ambiente,
    alpha=0.15,
    gamma=0.9,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.997,
    vitorias_necessarias=10, #ele precisa vencer 10 vezes consecutivas para afirmarmos que ele aprendeu o mundo.
    max_episodios=5000,
    q_tabela_inicial=None,
    verbose=True
):
    acoes   = [0, 1, 2, 3]
    q_tabela = {} if q_tabela_inicial is None else {k: v for k, v in q_tabela_inicial.items()}
    eps     = epsilon

    def obter_q(s, a):
        return q_tabela.get((s, a), 0.0)

    def escolher_acao(s, eps):
        if random.random() < eps:
            return random.choice(acoes)
        vals = [obter_q(s, a) for a in acoes]
        mx   = max(vals)
        return random.choice([a for a, v in zip(acoes, vals) if v == mx])

    episodio              = 0
    vitorias_consecutivas = 0
    historico_recompensas = [] 
    while vitorias_consecutivas < vitorias_necessarias and episodio < max_episodios:
        episodio += 1
        estado    = ambiente.reiniciar()
        encerrado = False
        passos    = 0
        recompensa_acumulada = 0 

        while not encerrado and passos < 200:
            passos += 1
            acao    = escolher_acao(estado, eps)
            prox_estado, recompensa, encerrado, vitoria = ambiente.passo(acao)

            recompensa_acumulada += recompensa 

            melhor_q_prox          = max(obter_q(prox_estado, a) for a in acoes)
            q_atual                = obter_q(estado, acao)
            q_tabela[(estado, acao)] = q_atual + alpha * (
                recompensa + gamma * melhor_q_prox - q_atual
            )
            estado = prox_estado

        eps = max(epsilon_min, eps * epsilon_decay)
        historico_recompensas.append(recompensa_acumulada)

        if encerrado and vitoria:
            vitorias_consecutivas += 1
        else:
            vitorias_consecutivas  = 0

        if verbose and episodio % 500 == 0:
            print(f"  Episódio {episodio:5d} | epsilon={eps:.4f} | "
                  f"vitórias consecutivas={vitorias_consecutivas}")

    convergiu = vitorias_consecutivas >= vitorias_necessarias
    return q_tabela, episodio, convergiu, historico_recompensas


def simular_agente(ambiente, q_tabela):
    acoes        = [0, 1, 2, 3]
    nomes_acoes  = {0: "Cima", 1: "Baixo", 2: "Esquerda", 3: "Direita"}
    estado       = ambiente.reiniciar()
    encerrado    = False
    caminho      = [ambiente.pos_agente]
    acoes_tomadas = []

    while not encerrado and len(caminho) <= 50:
        vals  = [q_tabela.get((estado, a), 0.0) for a in acoes]
        acao  = acoes[int(np.argmax(vals))]
        acoes_tomadas.append(nomes_acoes[acao])
        estado, _, encerrado, vitoria = ambiente.passo(acao)
        caminho.append(ambiente.pos_agente)

    if vitoria:
        print(f"  Atingiu o objetivo pelo caminho: {caminho}")
        print(f"  Ações:   {acoes_tomadas}")
    else:
        print(f" Não conseguiu atingir o objetivo e o caminho feito antes de parar foi: {caminho}")
    return vitoria


# Execução princial

random.seed(42)
np.random.seed(42)

# Primeiro mundo
print("-----------------------------")
print("Aprendendo o primeiro mundo")
print("-----------------------------")

pocos_1  = [(4, 4), (3, 3), (3, 1)]
wumpus_1 = (1, 3)
ouro_1   = (2, 3)
mundo1   = MundoWumpus(pocos=pocos_1, wumpus=wumpus_1, ouro=ouro_1)

mundo1.exibir_mapa()
print("Treinando o algoritmo no primeiro mundo")

q1, it1, conv1, hist1 = treinar_q_learning(
    mundo1,
    alpha=0.15,
    epsilon_decay=0.997,
    max_episodios=5000,
    verbose=True
)
print(f"\nConseguiu concluir: {conv1}  |  Episódios necessários: {it1}")
print("Simulação final no Mundo 1:")
simular_agente(mundo1, q1)

# Segundo mundo. Aqui temos transferência de aprendizado
print()
print("-----------------------------")
print("Aprendendo a andar no segundo mundo")
print("-----------------------------")

pocos_2  = [(2, 2), (1, 4)]
wumpus_2 = (4, 1)
ouro_2   = (4, 4)
mundo2   = MundoWumpus(pocos=pocos_2, wumpus=wumpus_2, ouro=ouro_2)

mundo2.exibir_mapa()
print("Tentando usar algo do aprendizado do mundo 1.")
print(" O epsilon foi reiniciado em 1.0")
print(" e o alpha aumentado para 0.5 (sobrescreve valores antigos mais rápido).")
print("A tabela Q do Mundo 1 foi usada como ponto de partida\n")


q2, it2, conv2, hist2 = treinar_q_learning(
    mundo2,
    alpha=0.5,
    epsilon_decay=0.999,
    max_episodios=5000,
    q_tabela_inicial=q1,
    verbose=True
)
print(f"\nConseguiu concluir: {conv2}  |  Episódios necessários: {it2}")
print("Simulação final no Mundo 2:")
simular_agente(mundo2, q2)

# Comparação
print()
print("-----------------------------")
print("Comparação")
print("-----------------------------")
print(f"  Mundo 1: {it1:5d} episódios  |  Conseguiu concluir: {conv1}")
print(f"  Mundo 2: {it2:5d} episódios  |  Conseguiu concluir: {conv2}")
print()
if conv1 and conv2:
    if it2 > it1:
        diff = it2 - it1
        print(f"  A conclusão é que tivemos transferência negativa.")
        print(f"  O agente concluiu nos dois mundos, mas precisou de {diff} episódios")
        print(f"  a mais no Mundo 2 ({it2} vs {it1}).")
        print(f"  O conhecimento prévio interferiu, mas foi superado pela re-exploração.")
    elif it2 < it1:
        print(f"  A conclusão é que a transferência foi positiva, pois o mundo 2 convergiu mais rápido.")
    else:
        print(f"Mesma velocidade de aprendizado.")

# Salvando arquivo JSON para criar gráficos 
dados_grafico = {
    "mundo1": hist1,
    "mundo2": hist2
}

with open("historico_recompensas.json", "w") as arquivo:
    json.dump(dados_grafico, arquivo)

print("Histórico de recompensas salvo em 'historico_recompensas.json'.")