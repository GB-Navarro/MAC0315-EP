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
        Return:
            None
    """

    def format_equation(coeffs: np.ndarray) -> str:
        """
            Description:
                Formata uma equação (função ou restrição) para exibição na forma padrão.
            Args:
                coeffs (np.ndarray): Coeficientes da equação.
            Return:
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
    
def preprocess_problem(c: np.ndarray, A: np.ndarray, relations: List[str], b: np.ndarray, 
                       decision_variables_limits: str) -> Tuple[str, np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    '''
        Description:
            Preprocessa o problema de otimização linear para adequá-lo ao formato padrão. Isso inclui a adição 
            de variáveis artificiais (e/ou variáveis de folga/excesso) e ajustes nas relações
            entre as restrições, para garantir que todas sejam do tipo igualdade (==).
            
        Args:
            problem_type (str): String indicando o tipo do problema ('max' ou 'min').
            c (np.ndarray): Vetor que contém os coeficientes de custo das variáveis de decisão.
            A (np.ndarray): Matriz que contém os coeficientes das restrições.
            relations (List[str]): Lista de strings que indicam as relações ('<=', '==', '>=') entre as restrições e os recursos.
            b (np.ndarray): Vetor que contém os valores dos recursos para cada restrição.
            decision_variables_limits (str): String que representa os limites das variáveis de decisão.

        Return:
            c (np.ndarray): Vetor de custos atualizado com as variáveis artificiais adicionadas.
            A (np.ndarray): Matriz de coeficientes das restrições, ajustada para o formato padrão.
            relations (List[str]): Lista atualizada de relações, convertendo todas para o tipo igualdade ('==').
            decision_variables_limits (str): Limites das variáveis de decisão.
    '''
    
    if all(elemento == "<=" for elemento in relations) or all(elemento == "==" for elemento in relations):
        # Caso todas as restrições sejam do tipo "<=" ou "==", adiciona variáveis de folga.
        m = b.shape[0] # Número de restrições
        A = np.hstack((A, np.eye(m))) # Adiciona variáveis de folga (matriz identidade).
        c = np.concatenate((c, np.zeros(m))) # Adiciona custos nulos para as variáveis de folga.
        relations = ["==" for _ in range(len(relations))] # Converte todas as relações para "==".
    
    if all(elemento == ">=" for elemento in relations):
        # Caso todas as restrições sejam do tipo ">=", adiciona variáveis de excesso e folga.
        m = b.shape[0] # Número de restrições
        A = np.hstack((A, -1*np.eye(m))) # Adiciona variáveis de excesso (matriz identidade negativa).
        A = np.hstack((A, np.eye(m))) # Adiciona variáveis artificiais (matriz identidade positiva).
        c = np.concatenate((c, np.zeros(2*m))) # Adiciona custos nulos para as novas variáveis.
        relations = ["==" for _ in range(len(relations))] # Converte todas as relações para "==".

    return c, A, relations, decision_variables_limits

def simplex_log(B_idx: list, N_idx: list, reduced_costs: np.ndarray, entering_index: int, leaving_index: int, verbose: bool) -> None:
    '''
        Description:
            Função auxiliar para registrar informações durante a execução do método simplex revisado.
            Quando o modo verbose está ativado, a função exibe os índices das variáveis básicas e não básicas,
            os custos reduzidos, e os índices das variáveis que estão entrando e saindo da base.
        Args:
            B_idx (list): Lista ou array contendo os índices das variáveis na base (B).
            N_idx (list): Lista ou array contendo os índices das variáveis fora da base (N).
            reduced_costs (np.ndarray): Array contendo os custos reduzidos das variáveis não básicas.
            entering_index (int): Índice da variável que está entrando na base.
            leaving_index (int): Índice da variável que está saindo da base.
            verbose (bool): Indica se as informações devem ser exibidas (True) ou não (False).
        Return:
            None
    '''
    
    if verbose:
        # Imprime os índices das variáveis básicas (B) e não básicas (N).
        print(f"B indices: {B_idx}, N indices: {N_idx}")
        
        # Imprime os custos reduzidos das variáveis não básicas.
        print(f"Reduced costs: {reduced_costs}")
        
        # Imprime o índice da variável que está entrando na base.
        print(f"Entering index: {entering_index}")
        
        # Imprime o índice da variável que está saindo da base.
        print(f"Leaving index: {leaving_index}\n")

def simplex_revised(problem_type: str, c: np.ndarray, A: np.ndarray, B_idx: list, N_idx: list, 
                    b: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, float]:
    '''
        Description:
            Implementa o método simplex revisado para resolver problemas de programação linear.
            O método busca iterativamente uma solução ótima movendo-se ao longo das arestas do politopo 
            viável até alcançar a solução ótima, de acordo com o tipo do problema (maximização ou minimização).
        Args:
            problem_type (str): Indica o tipo do problema, 'max' para maximização ou 'min' para minimização.
            c (np.ndarray): Vetor dos coeficientes da função objetivo (custos das variáveis de decisão).
            A (np.ndarray): Matriz das restrições (m linhas por n colunas).
            B_idx (list): Lista de índices das variáveis básicas iniciais.
            N_idx (list): Lista de índices das variáveis não básicas iniciais.
            b (np.ndarray): Vetor do lado direito das restrições (recursos disponíveis).
            verbose (bool, optional): Se True, exibe informações detalhadas sobre o progresso do algoritmo (o padrão é False).
        Return:
            Tuple:
                - x (np.ndarray): Vetor solução do problema (valores das variáveis de decisão).
                - z (float): Valor da função objetivo na solução ótima.
        Raises:
            ValueError: Caso o problema seja ilimitado (não há solução ótima finita).
    '''
        
    # Variável booleana que guardará o tipo do problema, isto é, maximização ou minimização.
    maximize = True if problem_type.lower() == 'max' else False
    
    # Obtem e armazena o número de restrições (m) e o número de variáveis de decisão (n).
    m, n = A.shape
    
    # Inicializa a matriz B com base nos índices da solução básica inicial (B_idx).
    B = A[:, B_idx]

    while True:
        # Realiza a fatoração LU da matriz B
        lu, piv = lu_factor(B)

        # Resolve o sistema linear B * x_B = b utilizando a decomposição LU da matriz B.
        x_B = lu_solve((lu, piv), b)

        # Computa os custos reduzidos.
        c_B = c[B_idx]
        c_N = c[N_idx]
        lambda_ = c_B.T @ lu_solve((lu, piv), np.eye(m))
        reduced_costs = c_N - lambda_ @ A[:, N_idx]

        # Verifica otimalidade.
        if maximize and np.all(reduced_costs <= 1e-8): 
            x = np.zeros(n)
            x[B_idx] = x_B
            z = c.T @ x
            return x, z
        elif not maximize and np.all(reduced_costs >= -1e-8):
            x = np.zeros(n)
            x[B_idx] = x_B
            z = c.T @ x  
            return x, z

        # Escolhe uma variável para entrar na base.
        if maximize:
            entering_idx = N_idx[np.argmax(reduced_costs)]
        else:
            entering_idx = N_idx[np.argmin(reduced_costs)]
        
        # Computa a direção p = B^-1 * A_j.
        p = lu_solve((lu, piv), A[:, entering_idx])

        if np.all(p <= 0):
            # Exibe um erro indicando que o problema em questão é ilimitado.
            raise ValueError("Problema ilimitado.")

        # Realiza a regra de razão mínima para saber quem sai da base (com supressão de eventuais warnings por haver 0's em p).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ratios = np.where(p > 0, x_B / p, np.inf)
        leaving_idx = B_idx[np.argmin(ratios)]

        # Caso verbose seja igual à True, exibe um log com algumas informações da iteração atual do método.
        simplex_log(B_idx, N_idx, reduced_costs, entering_idx, leaving_idx, verbose)
        
        # Atualiza a base.
        B_idx[B_idx.index(leaving_idx)] = entering_idx
        N_idx[N_idx.index(entering_idx)] = leaving_idx
        B = A[:, B_idx]
        
def simplex_revised_two_phases(problem_type, c, A, b, relations):
    '''
        Description:

        Args:

        Return:

    '''
    
    # Passo 1: Converter desigualdades em igualdades e adicionar variáveis artificiais
    m, n = A.shape
    
    phase1_N_idx = list(range(n - m))
    c_phase1 = np.zeros(n-m)
    c_phase1 = np.concatenate((c_phase1, np.ones(m)))

    # Passo 2: Resolver Fase 1 para minimizar soma das variáveis artificiais
    x_phase1, z_phase1 = simplex_revised('min',c_phase1, A, list(range(n - m, n)), phase1_N_idx,  b, True)
    
    B_idx = list(np.nonzero(x_phase1)[0])
    N_idx = list(np.setdiff1d(list(range(n-m)), B_idx)) # (ESSE N_idx PODE SER CALCULADO DENTRO DA FUNÇÃO 'simplex_revised'!)

    if z_phase1 > 1e-6:
        raise ValueError("Problema inviável: as variáveis artificiais não podem ser eliminadas.")
    
    # Remover colunas correspondentes às variáveis artificiais
    A = np.delete(A, list(range(n - m, n)), axis=1)
    c = np.delete(c, list(range(n - m, n)))
    
    # Passo 3: Resolver Fase 2 com a função objetivo original
    x_phase2, z_phase2 = simplex_revised(problem_type, c, A, B_idx, N_idx, b, True)

    return x_phase2, z_phase2

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
            problem_type, c, A, original_relations, b, decision_variables_limits = process_entry(content)
            
            #
            print("O problema a ser resolvido é:\n")
            show_problem(problem_type, c, A, original_relations, b, decision_variables_limits)
            
            #
            c, A, relations, decision_variables_limits = preprocess_problem(c, A, original_relations, b, decision_variables_limits)

            print("Quando transformado para a forma padrão, o problema em questão se torna:\n")
            show_problem(problem_type, c, A, relations, b, decision_variables_limits)

            if all(relation == "<=" for relation in original_relations):
                m, n = A.shape
                B_idx = list(range(n - m, n))
                N_idx = list(range(n - m))
                x, z = simplex_revised(problem_type, c, A, B_idx, N_idx, b, True)
                print(f"Solução ótima: {x}")
                print(f"Valor ótimo: {z}")
                
            elif all(relation == ">=" for relation in original_relations) or all(relation == "==" for relation in original_relations):
                x,z = simplex_revised_two_phases(problem_type,c,A,b,relations)
                print(f"Solução ótima: {x}")
                print(f"Valor ótimo: {z}")
                
    except FileNotFoundError:
        print("Arquivo não encontrado")
        
    except ValueError as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()