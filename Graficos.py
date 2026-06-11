import json
import matplotlib.pyplot as plt
import numpy as np

def calcular_media_movel(dados_brutos, tamanho_janela=50):
    """
    Calcula a média móvel para suavizar a curva do gráfico.
    """
    if len(dados_brutos) < tamanho_janela:
        return dados_brutos
    pesos_janela = np.ones(tamanho_janela) / tamanho_janela
    recompensas_suavizadas = np.convolve(dados_brutos, pesos_janela, mode='valid')
    return recompensas_suavizadas

def gerar_graficos_wumpus(caminho_arquivo="historico_recompensas.json"):
    # Carregamento dos dados
    try:
        with open(caminho_arquivo, "r") as arquivo:
            dados = json.load(arquivo)
    except FileNotFoundError:
        print(f"O arquivo '{caminho_arquivo}' não foi encontrado.")
        return

    historico_mundo1 = dados.get("mundo1", [])
    historico_mundo2 = dados.get("mundo2", [])

    # Calcula a média 
    tamanho_janela_suavizacao = 50
    mundo1_suavizado = calcular_media_movel(historico_mundo1, tamanho_janela_suavizacao)
    mundo2_suavizado = calcular_media_movel(historico_mundo2, tamanho_janela_suavizacao)

    # Configura a figura 
    plt.style.use('seaborn-v0_8-whitegrid') 
    figura, eixos = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico do Mundo 1
    eixos[0].plot(historico_mundo1, color='blue', alpha=0.15, label='Dados Brutos')
    eixos[0].plot(range(tamanho_janela_suavizacao - 1, len(historico_mundo1)), 
                  mundo1_suavizado, color='darkblue', linewidth=2, label='Média Móvel')
    
    eixos[0].set_title("Curva de Aprendizado - Mundo 1", fontsize=14)
    eixos[0].set_xlabel("Episódios", fontsize=12)
    eixos[0].set_ylabel("Recompensa Total", fontsize=12)
    eixos[0].legend()

    # Gráfico do Mundo 2
    eixos[1].plot(historico_mundo2, color='orange', alpha=0.2, label='Dados Brutos')
    eixos[1].plot(range(tamanho_janela_suavizacao - 1, len(historico_mundo2)), 
                  mundo2_suavizado, color='darkorange', linewidth=2, label='Média Móvel')
    
    eixos[1].set_title("Curva de Aprendizado - Mundo 2", fontsize=14)
    eixos[1].set_xlabel("Episódios", fontsize=12)
    eixos[1].set_ylabel("Recompensa Total", fontsize=12)
    eixos[1].legend()

    plt.tight_layout()

    # Salvar e exibir
    nome_imagem = "grafico_transferencia.png"
    plt.savefig(nome_imagem, dpi=300, bbox_inches='tight')
    
    print(f"[+] Gráfico gerado salvo como '{nome_imagem}'!")
    plt.show()

if __name__ == "__main__":
    gerar_graficos_wumpus()