import sys
import numpy as np
from typing import Tuple, List

def process_entry(content: str) -> Tuple[str, np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    '''
        Description:
            Processa o conteúdo de uma string, visando extrair as informações de um problema de otimização linear.
        Args:
            content (str): Texto contendo a descrição do problema.
        Return:
            problem_type (str): String (max || min) que representa o tipo do problema, isto é, maximização ou minimização.
            c (np.ndarray): Vetor que contém os coeficientes que representam os custos de cada uma das n variáveis de decisão.
            A (np.ndarray): Matriz que contém os coeficientes de cada uma das m restrições.
            relations (list): Lista que contém as relações entre cada uma das restrições e seus respectivos recursos.
            b (np.ndarray): Vetor que contém o valor de cada um dos recursos...
            decision_variables_limits (str): String que representa o limite das variáveis de decisão
    '''

    # Remove linhas vazias e comentários.
    lines = [line for line in content.split("\n") if (line != '' and not line.startswith('#'))]

    # Armazena a string que indica o tipo de problema.
    problem_type = lines[0]

    # Armazena o vetor que contém os coeficientes que representam os custos de cada uma das variáveis de decisão.
    c = np.fromstring(lines[1],sep=" ")
    # Armazena o número de variáveis de decisão.
    decision_variables_number = c.shape[0]

    # Armazena a string que indica os limites das variáveis de decisão.
    decision_variables_limits = lines[-1]
    
    # Cria algumas variáveis para processar as restrições.
    restrictions = lines[2:len(lines)-1] # Lista de strings que representam todas as restrições do problema.
    A = [] # Variável que armazenará a matriz que contém os coeficientes de cada uma das restrições.
    b = [] # Variável que armazenará o vetor de recursos.
    relations = [] # Variável que armazenará as relações (<=, ==, >=) entre as restrições e seus respectivos recursos. 
   
    for restriction in restrictions:
        # Obtem e armazena a relação (<=, ==, >=) entre a restrição e o recurso em questão.
        relation = restriction[decision_variables_number*2:decision_variables_number*2+2]
        relations.append(relation)
        
        # Divide a string 'restriction' em dois elementos:
        # - A parte à esquerda da relação (posição 0) contém os coeficientes da linha em questão da matriz A .
        # - A parte à direita da relação (posição 1) contém o recurso correspondente a linha em questão (valor de b).
        restriction = restriction.split(f" {relation} ")
        
        # Adiciona uma nova linha de coeficientes a matriz A.
        A.append(np.fromstring(restriction[0],sep=" "))
        
        # Adiciona um novo coeficiente de recurso ao vetor b.
        b.append(np.float64(restriction[1]))
    
    # Transforma tanto a matriz A quanto o vetor b em objetos do tipo np.ndarray (numpy array), visando facilitar a manipulação dos mesmos.  
    A = np.array(A)
    b = np.array(b)

    return problem_type, c, A, relations, b, decision_variables_limits

def main():
    
    if len(sys.argv) < 2:
        print("Forneça o caminho do arquivo!")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            problem_type, c, A, delimitators, b, decision_variables_limits = process_entry(content)
    except FileNotFoundError:
        print("Arquivo não encontrado")
    except ValueError as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()