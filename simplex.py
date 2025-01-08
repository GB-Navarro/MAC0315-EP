#
import re
#
import sys
#
import warnings
#
import numpy as np
#
from typing import Tuple, List
#
from scipy.linalg import lu_factor, lu_solve # Biblioteca temporária


def process_entry(content: str) -> Tuple[str, np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    '''
        Description:
            Processa o conteúdo do arquivo de entrada, visando extrair as informações de um problema de otimização linear.
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
    
    # Armazena a string que indica os limites das variáveis de decisão.
    decision_variables_limits = lines[-1]

    # Cria algumas variáveis para processar as restrições.
    restrictions = lines[2:len(lines)-1] # Lista de strings que representam todas as restrições do problema.
    A = [] # Variável que armazenará a matriz que contém os coeficientes de cada uma das restrições.
    b = [] # Variável que armazenará o vetor de recursos.
    relations = [] # Variável que armazenará as relações (<=, ==, >=) entre as restrições e seus respectivos recursos. 

    for restriction in restrictions:
        # Divide a string 'restriction' em três partes usando os operadores (<=, ==, >=) como delimitadores:
        # - Posição 0: Coeficientes da linha correspondente da matriz A.
        # - Posição 1: Relação entre os coeficientes e o recurso (<=, ==, >=).
        # - Posição 2: Valor do recurso correspondente (elemento de b).
        restriction = re.split(r"( <= | == | >= )",restriction)
        
        # Obtem e armazena a relação (<=, ==, >=) entre a restrição e o recurso em questão.
        relation = restriction[1].strip()
        relations.append(relation)
        
        # Converte os coeficientes (posição 0) em um array numérico e adiciona à matriz A.
        A.append(np.fromstring(restriction[0],sep=" "))
        
        # Converte o recurso (posição 2) em um número e adiciona ao vetor b.
        b.append(np.float64(restriction[2]))
    
    # Transforma tanto a matriz A quanto o vetor b em objetos do tipo np.ndarray (numpy array), visando facilitar a manipulação dos mesmos.  
    A = np.array(A)
    b = np.array(b)

    return problem_type, c, A, relations, b, decision_variables_limits

def show_problem(problem_type: str, c: np.ndarray, A: np.ndarray, relations: List[str], b: np.ndarray, decision_variables_limits: str) -> None:
    """
        Description:
            Exibe o problema de programação linear na forma padrão.
        Args:
            problem_type (str): Tipo do problema ('max' para maximização ou 'min' para minimização).
            c (np.ndarray): Vetor de coeficientes da função objetivo.
            A (np.ndarray): Matriz de coeficientes das restrições.
            relations (List[str]): Lista de relações das restrições (e.g., '<=', '>=' ou '=').
            b (np.ndarray): Vetor de valores dos recursos.
            decision_variables_limits (str): Limites das variáveis de decisão.
        Returns:
            None
    """

    def format_equation(coeffs: np.ndarray) -> str:
        """
            Description:
                Formata uma equação (função ou restrição) para exibição na forma padrão.
            Args:
                coeffs (np.ndarray): Coeficientes da equação.
            Returns:
                str: Equação formatada.
        """

        return " ".join(f"{'-' if coeff < 0 else ('+' if (coeff >= 0 and idx > 0) else '')} {abs(coeff):.2f}x{idx + 1}" for idx, coeff in enumerate(coeffs)).strip()

    # Exibe o tipo do problema
    problem_type_text = "Maximizar" if problem_type.lower() == "max" else "Minimizar"
    print(f"{problem_type_text} Z = {format_equation(c)}\n")

    # Exibe as restrições
    print("Sujeito a:")
    for coeffs, relation, resource in zip(A, relations, b):
        print(f"\t{format_equation(coeffs)} {relation} {resource:.2f}")

    # Exibe os limites das variáveis de decisão
    print(f"\t{decision_variables_limits}\n")
    
def preprocess_problem(problem_type: str, c: np.ndarray, A: np.ndarray, relations: List[str], b: np.ndarray, decision_variables_limits: str) -> Tuple[str, np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    '''
        Description:

        Args:

        Return:
    '''
    
    if all(elemento == "<=" for elemento in relations) or all(elemento == "==" for elemento in relations):
        #
        m = b.shape[0]
        #
        A = np.hstack((A, np.eye(m)))
        #
        c = np.concatenate((c, np.zeros(m)))
        #
        relations = ["==" for _ in range(len(relations))]
    
    if all(elemento == ">=" for elemento in relations):
        #
        m = b.shape[0]
        #
        A = np.hstack((A, -1*np.eye(m)))
        #
        A = np.hstack((A, np.eye(m)))
        #
        c = np.concatenate((c, np.zeros(2*m)))
        #
        relations = ["==" for _ in range(len(relations))]

    return problem_type, c, A, relations, b, decision_variables_limits

def simplex_revised(c, A, b, verbose=False):
    '''
        Description:

        Args:

        Return:

    '''
    m, n = A.shape
    B_idx = list(range(n - m, n))
    N_idx = list(range(n - m))

    # Inicializar B
    B = A[:, B_idx]

    if verbose:
        print("------------------------------------------- Depuração -------------------------------------------\n")
        
    while True:
        if verbose:
            print(f"B indices: {B_idx}, N indices: {N_idx}")

        # Fatoração LU de B
        lu, piv = lu_factor(B)

        # Resolve B * x_B = b
        x_B = lu_solve((lu, piv), b)

        # Computa custos reduzidos
        c_B = c[B_idx]
        c_N = c[N_idx]
        lambda_ = c_B.T @ lu_solve((lu, piv), np.eye(m))
        reduced_costs = c_N - lambda_ @ A[:, N_idx]

        if verbose:
            print(f"Reduced costs: {reduced_costs}\n")

        # Verifica otimalidade
        if np.all(reduced_costs <= 1e-8):
            x = np.zeros(n)
            x[B_idx] = x_B
            z = c.T @ x
            
            if verbose:
                print("------------------------------------------- --------- -------------------------------------------\n")
        
            return x, z

        # Escolhe uma variável para entrar na base
        entering_idx = N_idx[np.argmax(reduced_costs)]
        
        # Computa a direção p = B^-1 * A_j
        p = lu_solve((lu, piv), A[:, entering_idx])

        if np.all(p <= 0):
            if verbose:
                print("------------------------------------------- --------- -------------------------------------------\n")
                
            raise ValueError("Problema ilimitado.")

        # Regra de razão mínima com supressão de warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ratios = np.where(p > 0, x_B / p, np.inf)
            
        leaving_idx = B_idx[np.argmin(ratios)]

        # Atualiza a base
        B_idx[B_idx.index(leaving_idx)] = entering_idx
        N_idx[N_idx.index(entering_idx)] = leaving_idx
        B = A[:, B_idx]

def main():
    
    if len(sys.argv) < 2:
        print("Forneça o caminho do arquivo!")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    try:
        with open(file_path, 'r') as file:
            #
            content = file.read()
            #
            problem_type, c, A, delimitators, b, decision_variables_limits = process_entry(content)
            
            #
            print("O problema a ser resolvido é:\n")
            show_problem(problem_type, c, A, delimitators, b, decision_variables_limits)
            
            #
            problem_type, c, A, delimitators, b, decision_variables_limits = preprocess_problem(problem_type, c, A, 
                                                                                                delimitators, b, decision_variables_limits)

            print("Quando transformado para a forma padrão, o problema em questão se torna:\n")
            show_problem(problem_type, c, A, delimitators, b, decision_variables_limits)
            
            print(f"A\n{A}")
            print(f"b\n{b}")
            #x, z = simplex_revised(c, A, b, True)
            #print(f"Solução ótima: {x}")
            #print(f"Valor ótimo: {z}")
            
    except FileNotFoundError:
        print("Arquivo não encontrado")
        
    except ValueError as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()