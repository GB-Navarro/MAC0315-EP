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

def process_entry(content: str) -> Tuple[str, np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    '''
        Description:
            Processa o conteúdo do arquivo de entrada, visando extrair as informações de um problema de otimização linear.
        Args:
            content (str): Texto contendo a descrição do problema.
        Return:
            problem_type (str): String (max || min) que representa o tipo do problema, isto é, maximização ou minimização.
            c (np.ndarray): Vetor que contém os coeficientes que representam os custos de cada uma das n variáveis de decisão.
            variables (list):
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

    #
    variables = [f"x{i+1}" for i in range(len(c))]
    
    return problem_type, c, variables, A, relations, b, decision_variables_limits

def show_problem(problem_type: str, c: np.ndarray, variables: list, A: np.ndarray, relations: List[str], b: np.ndarray, decision_variables_limits: str) -> None:
    """
        Description:
            Exibe o problema de programação linear na forma padrão.
        Args:
            problem_type (str): Tipo do problema ('max' para maximização ou 'min' para minimização).
            c (np.ndarray): Vetor de coeficientes da função objetivo.
            variables (list):
            A (np.ndarray): Matriz de coeficientes das restrições.
            relations (List[str]): Lista de relações das restrições (e.g., '<=', '>=' ou '=').
            b (np.ndarray): Vetor de valores dos recursos.
            decision_variables_limits (str): Limites das variáveis de decisão.
        Return:
            None
    """

    def format_equation(coeffs: np.ndarray, variables: list) -> str:
        """
            Description:
                Formata uma equação (função ou restrição) para exibição na forma padrão.
            Args:
                coeffs (np.ndarray): Coeficientes da equação.
                variables (list):
            Return:
                str: Equação formatada.
        """

        return " ".join(f"{'-' if coeff < 0 else ('+' if (coeff >= 0 and idx > 0) else '')} {abs(coeff):.2f}{variables[idx]}" for idx, coeff in enumerate(coeffs)).strip()

    # Exibe o tipo do problema
    problem_type_text = "Maximizar" if problem_type.lower() == "max" else "Minimizar"
    print(f"{problem_type_text} Z = {format_equation(c, variables)}\n")

    # Exibe as restrições
    print("Sujeito a:")
    for coeffs, relation, resource in zip(A, relations, b):
        tab = '\t' if coeffs[0] >= 0 else '      '
        print(f"{tab}{format_equation(coeffs, variables)} {relation} {resource:.2f}")


    # Exibe os limites das variáveis de decisão
    print(f"\t{decision_variables_limits}\n")
    
#def preprocess_problem(c: np.ndarray, A: np.ndarray, relations: List[str], b: np.ndarray, 
#                       decision_variables_limits: str) -> Tuple[str, np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    '''
        Description:
            Preprocessa o problema de otimização linear para adequá-lo ao formato padrão. Isso inclui a adição 
            de variáveis artificiais (e/ou variáveis de folga/excesso) e ajustes nas relações
            entre as restrições, para garantir que todas sejam do tipo igualdade (==).
            
        Args:
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
    '''
    
def preprocess_problem(
    c: np.ndarray, 
    variables: list,
    A: np.ndarray, 
    relations: List[str], 
    b: np.ndarray, 
    decision_variables_limits: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, bool]:
    """
    Description:
        Preprocessa o problema de programação linear para adequá-lo ao formato padrão.
        Trata limites das variáveis de decisão, converte restrições para igualdades,
        adiciona variáveis de folga/excesso e identifica a necessidade do método das duas fases.

    Args:
        c (np.ndarray): Vetor de custos das variáveis de decisão.
        variables (list):
        A (np.ndarray): Matriz de coeficientes das restrições.
        relations (List[str]): Lista indicando as relações ('<=', '==', '>=') entre as restrições e b.
        b (np.ndarray): Vetor de recursos disponíveis para as restrições.
        decision_variables_limits (List[Tuple[float, float]]): Limites das variáveis de decisão (i, j) onde i ≤ x ≤ j.

    Returns:
        Tuple:
            - c (np.ndarray): Vetor de custos atualizado com as novas variáveis.
            - variables (list):
            - A (np.ndarray): Matriz de coeficientes ajustada.
            - relations (list[str]): Lista de relações convertida para igualdades.
            - b (np.ndarray): Vetor de recursos ajustado.
            - artificial_variables_columns_indexes (list):
            - slack_variables_columns_indexes (list):
            - requires_two_phases (bool): Indica se o método das duas fases será necessário.
    """
    n = A.shape[1]  # Número de restrições e variáveis
    
    #
    if decision_variables_limits != 'xi >= 0':
        lower_value, upper_value = np.float64(decision_variables_limits.split(" <= xi <= ")[0]), np.float64(decision_variables_limits.split(" <= xi <= ")[1])
        
        for i in range(n):
            #
            if not np.isinf(lower_value):  
                new_row = np.zeros(n)
                new_row[i] = 1  # -x_i >= -lower
                A = np.vstack([A, new_row])
                b = np.append(b, lower_value)
                relations.append(">=")
            #    
            if not np.isinf(upper_value):
                new_row = np.zeros(n)
                new_row[i] = 1  # x_i <= upper
                A = np.vstack([A, new_row])
                b = np.append(b, upper_value)
                relations.append("<=")
    
    # 2. Verificar e ajustar restrições com recursos negativos
    for i in range(len(b)):
        if b[i] < 0:
            A[i, :] *= -1
            b[i] *= -1
            if relations[i] == "<=":
                relations[i] = ">="
            elif relations[i] == ">=":
                relations[i] = "<="
    
    #
    artificial_variables_rows_indexes = [index for index in range(len(relations)) if (relations[index] == "==" or relations[index] == ">=")]
    #
    artificial_variables_columns_indexes = []
    #
    requires_two_phases = True if len(artificial_variables_rows_indexes) > 0 else False
    
    # 3. Adicionar variáveis de folga e excesso 
    slack_variables_rows_indexes = []
    #
    slack_variables_columns_indexes = []
    #
    slack_variable_index_count = 1
    excess_variable_index_count = 1
    for i, relation in enumerate(relations):
        if relation == "<=":  # Adicionar variável de folga
            slack_variable = np.zeros((A.shape[0], 1))
            slack_variable[i, 0] = 1
            slack_variables_columns_indexes.append(A.shape[1])
            A = np.hstack([A, slack_variable])
            c = np.append(c, 0)
            relations[i] = "=="
            variables.append(f"s{slack_variable_index_count}")
            slack_variable_index_count += 1
            slack_variables_rows_indexes.append(i)
        elif relation == ">=":  # Adicionar variável de excesso
            excess_variable = np.zeros((A.shape[0], 1))
            excess_variable[i, 0] = -1
            A = np.hstack([A, excess_variable])
            c = np.append(c, 0)
            relations[i] = "=="
            variables.append(f"e{excess_variable_index_count}")
            excess_variable_index_count += 1
    
    # 4. Adicionar variáveis artificiais apenas se necessário
    if requires_two_phases:
        #
        artificial_variable_index_count = 1
        for index in artificial_variables_rows_indexes:
            artificial_variable = np.zeros((A.shape[0],1))
            artificial_variable[index,0] = 1
            artificial_variables_columns_indexes.append(A.shape[1])
            A = np.hstack([A, artificial_variable])
            c = np.append(c, 0)
            variables.append(f"a{artificial_variable_index_count}")
            artificial_variable_index_count += 1
    
    return c, A, relations, b, decision_variables_limits, artificial_variables_columns_indexes, slack_variables_columns_indexes, requires_two_phases


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

def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Description:
            Realiza a decomposição LU de uma matriz A com pivotamento parcial. 
            A decomposição LU fatoriza a matriz A em três componentes: 
            uma matriz de permutação P, uma matriz triangular inferior L e uma matriz triangular superior U,
            tal que P @ A = L @ U. O pivotamento parcial é utilizado para evitar problemas de estabilidade
            numérica em matrizes mal condicionadas ou com pivôs zero.
        Args:
            A (np.ndarray): Matriz quadrada a ser fatorada (dimensão n x n).
        Return:
            Tuple:
                - P (np.ndarray): Matriz de permutação (n x n), que indica as trocas de linhas realizadas.
                - L (np.ndarray): Matriz triangular inferior (n x n), com 1's na diagonal principal.
                - U (np.ndarray): Matriz triangular superior (n x n), obtida após a eliminação Gaussiana.
        Raises:
            ValueError: Caso o pivotamento parcial não consiga evitar um pivô zero, impossibilitando a decomposição.
        Notes:
            - Este método assume que A é uma matriz quadrada. Caso contrário, deve ser adaptado.
            - A matriz P é tal que o produto P @ A reordena as linhas de A para permitir o maior pivô possível
              em cada etapa.
    """
    
    n = A.shape[0] # Obtém a dimensão da matriz quadrada A (n x n).
    
    # Inicializa as matrizes:
    # P: Matriz de permutação como a matriz identidade (sem trocas iniciais).
    # L: Matriz triangular inferior zerada (os fatores de eliminação serão preenchidos posteriormente).
    # U: Cópia da matriz A, que será transformada em uma matriz triangular superior.
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    # Realiza a eliminação Gaussiana com pivotamento parcial.
    for i in range(n):
        # Pivotamento parcial: Identifica a linha com o maior valor absoluto na coluna atual (i)
        # para evitar divisões por pivôs muito pequenos ou zero.
        max_row = np.argmax(np.abs(U[i:, i])) + i
        if i != max_row:
            # Troca as linhas i e max_row em U e P.
            U[[i, max_row]] = U[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
            # Garante que L também seja atualizado para refletir a troca de linhas.
            L[[i, max_row], :i] = L[[max_row, i], :i]

        # Verifica se o pivô (U[i, i]) é zero após o pivotamento.
        # Se for zero, a matriz é singular e a decomposição não pode continuar.
        if U[i, i] == 0:
            raise ValueError("Decomposição LU não é possível devido a pivô zero, mesmo após pivotamento.")

        # Eliminação de Gauss:
        # Para cada linha abaixo da linha atual (i), calcula o fator de eliminação (L[j, i]) 
        # e atualiza a matriz U.
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i] # Fator de eliminação para a linha j.
            L[j, i] = factor # Armazena o fator de eliminação na matriz L.
            U[j, i:] -= factor * U[i, i:] # Atualiza a linha j de U.

    # Garante que a diagonal principal de L seja composta por 1's.
    np.fill_diagonal(L, 1)
    
    return P, L, U

def lu_solve(P: np.ndarray, L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
        Description:
            Resolve o sistema linear Ax = b utilizando a decomposição LU com pivotamento parcial.
            A solução é obtida em três etapas:
            1. Aplica-se a matriz de permutação P ao vetor b, resultando no vetor permutado Pb.
            2. Realiza-se a substituição para frente, resolvendo o sistema Ly = Pb.
            3. Realiza-se a substituição para trás, resolvendo o sistema Ux = y.
        Args:
            P (np.ndarray): Matriz de permutação (n x n), obtida durante a decomposição LU.
            L (np.ndarray): Matriz triangular inferior (n x n), com 1's na diagonal principal.
            U (np.ndarray): Matriz triangular superior (n x n), obtida durante a decomposição LU.
            b (np.ndarray): Vetor do lado direito do sistema (dimensão n).
        Return:
            x (np.ndarray): Solução do sistema linear (dimensão n), tal que Ax = b.
        Notes:
            - Este método é usado após realizar a decomposição LU de uma matriz A. 
              Ele assume que as matrizes P, L e U estão corretamente fatoradas.
            - A matriz A original é tal que P @ A = L @ U.
            - Para evitar erros numéricos, a matriz U não deve ter valores muito próximos de zero na diagonal.
        Complexity:
            - Aplicação da permutação: O(n^2) para multiplicar P por b.
            - Substituição para frente e para trás: O(n^2) cada, devido à natureza triangular das matrizes.
        Raises:
            ValueError: Caso algum elemento da diagonal de U seja zero, indicando que o sistema é singular
            e não pode ser resolvido.
    """
    
    # Aplica a matriz de permutação ao vetor b:
    # - Reorganiza os elementos de b de acordo com as trocas realizadas durante o pivotamento parcial.
    b_permuted = P @ b

    # Substituição para frente:
    # - Resolve Ly = Pb, utilizando a propriedade triangular inferior de L.
    # - Calcula y[i] como o valor de b_permuted[i] menos a soma dos produtos dos elementos de L com os valores previamente calculados em y.
    n = L.shape[0]
    y = np.zeros_like(b_permuted, dtype=float)
    for i in range(n):
        y[i] = b_permuted[i] - np.dot(L[i, :i], y[:i])

    # Substituição para trás:
    # - Resolve Ux = y, utilizando a propriedade triangular superior de U.
    # - Calcula x[i] como o valor de y[i] menos a soma dos produtos dos elementos de U
    #   com os valores previamente calculados em x, dividido pelo pivô U[i, i].
    x = np.zeros_like(b_permuted, dtype=float)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Sistema linear singular: pivô zero encontrado em U.")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

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
        P, L, U = lu_decomposition(B)

        # Resolve o sistema linear B * x_B = b utilizando a decomposição LU da matriz B.
        x_B = lu_solve(P, L, U, b)

        # Computa os custos reduzidos.
        c_B = c[B_idx]
        c_N = c[N_idx]
        lambda_ = c_B.T @ lu_solve(P, L, U, np.eye(m))
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
        p = lu_solve(P, L, U, A[:, entering_idx])

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
        
def simplex_revised_two_phases(problem_type: str, c: np.ndarray, A: np.ndarray, b: np.ndarray, B_idx_phase1: list,
                               N_idx_phase1: list, variables: list, verbose: bool = False) -> Tuple[np.ndarray, float]:
    '''
        Description:
            Implementa o método simplex revisado em duas fases para resolver problemas de programação linear.
            Este método é utilizado quando o problema possui desigualdades que precisam ser convertidas em igualdades
            através da adição de variáveis artificiais. A Fase 1 busca uma solução inicial viável minimizando a soma
            das variáveis artificiais, e a Fase 2 resolve o problema original.
        Args:
            problem_type (str): Tipo do problema ('max' para maximização ou 'min' para minimização).
            c (np.ndarray): Vetor dos coeficientes da função objetivo original.
            A (np.ndarray): Matriz das restrições (m linhas por n colunas).
            b (np.ndarray): Vetor do lado direito das restrições (recursos disponíveis).
            B_idx_phase1 (list):
            N_idx_phase1 (list):
            variables (list):
            verbose (bool, optional): Indica se informações detalhadas sobre o progresso do algoritmo devem ser exibidas (o padrão é False).
        Return:
            Tuple:
                - x_phase2 (np.ndarray): Vetor solução do problema original (valores das variáveis de decisão).
                - z_phase2 (float): Valor da função objetivo na solução ótima do problema original.
        Raises:
            ValueError: Caso o problema seja inviável (não há solução viável) ou ilimitado (não há solução ótima finita).
    '''
    
    #
    non_artificial_variables_number = len([variable for variable in variables if variable[0] != "a"])
    artificial_variables_number = A.shape[1] - non_artificial_variables_number

    # Cria o vetor de custos para a Fase 1:
    # - Primeiros (n-m) elementos correspondem a zeros (variáveis originais do problema).
    # - Últimos m elementos correspondem a 1's (variáveis artificiais que serão minimizadas).
    c_phase1 = np.zeros(non_artificial_variables_number)
    c_phase1 = np.concatenate((c_phase1, np.ones(artificial_variables_number)))

    # Resolve a Fase 1 para minimizar a soma das variáveis artificiais:
    # - range(n - m, n): Índices das variáveis artificiais (inicialmente na base).
    # - range(n - m): Índices das variáveis originais (inicialmente não básicas).
    x_phase1, z_phase1 = simplex_revised('min', c_phase1, A, B_idx_phase1, N_idx_phase1, b, verbose)
    
    # Atualiza os índices das variáveis básicas (B_idx) e não básicas (N_idx):
    # - B_idx: Índices das variáveis que possuem valor não nulo na solução inicial.
    # - N_idx: Índices das variáveis restantes (não básicas).
    B_idx = list(np.nonzero(x_phase1)[0])
    N_idx = list(np.setdiff1d(list(range(non_artificial_variables_number)), B_idx))

    # Verifica a viabilidade do problema:
    # - Se a soma das variáveis artificiais (z_phase1) for maior que um pequeno limite tolerável (~0),
    #   isso indica que o problema é inviável, pois a solução viável inicial não existe.
    if z_phase1 > 1e-6:
        raise ValueError("Problema inviável: as variáveis artificiais não podem ser eliminadas.")
    
    # Remove as colunas correspondentes às variáveis artificiais:
    # - As colunas associadas às variáveis artificiais (índices de n-m a n) são removidas da matriz A.
    # - Os custos correspondentes no vetor c também são eliminados.
    A = np.delete(A, list(range(non_artificial_variables_number, non_artificial_variables_number + artificial_variables_number)), axis=1)
    c = np.delete(c, list(range(non_artificial_variables_number, non_artificial_variables_number + artificial_variables_number)))
    
    # Resolve a Fase 2 para encontrar a solução ótima do problema original:
    # - Utiliza a solução viável inicial encontrada na Fase 1 (B_idx, N_idx) como ponto de partida.
    x_phase2, z_phase2 = simplex_revised(problem_type, c, A, B_idx, N_idx, b, verbose)

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
            problem_type, c, variables, A, original_relations, b, decision_variables_limits = process_entry(content)
            
            #
            decision_variables_number = len(c)
            
            #
            print("O problema a ser resolvido é:\n")
            show_problem(problem_type, c, variables, A, original_relations, b, decision_variables_limits)
            
            #
            c, A, relations, b, decision_variables_limits, artificial_variables_columns_indexes, \
            slack_variables_columns_indexes, requires_two_phases = preprocess_problem(c, variables, A, original_relations, b, decision_variables_limits)

            print("Quando transformado para a forma padrão, o problema em questão se torna:\n")
            show_problem(problem_type, c, variables, A, relations, b, decision_variables_limits)

            if requires_two_phases:
                n = A.shape[1]
                print("\nPara resolver o problema em questão, usaremos o simplex em duas fases.\n")
                B_idx = artificial_variables_columns_indexes + slack_variables_columns_indexes
                N_idx = list(np.setdiff1d(list(range(n)), B_idx))
                x,z = simplex_revised_two_phases(problem_type, c, A, b, B_idx, N_idx, variables)
                print(f"Solução ótima: {' '.join([f'{var} = {coeff}' for var, coeff in zip(variables[:decision_variables_number], x[:decision_variables_number])])}")
                print(f"\nValor ótimo:   Z = {z}")
            else:
                n = A.shape[1]
                B_idx = slack_variables_columns_indexes
                N_idx = list(np.setdiff1d(list(range(n)), B_idx))
                x, z = simplex_revised(problem_type, c, A, B_idx, N_idx, b)
                print(f"Solução ótima: {' '.join([f'{var} = {coeff}' for var, coeff in zip(variables[:decision_variables_number], x[:decision_variables_number])])}")
                print(f"\nValor ótimo: {z}")
            
    except FileNotFoundError:
        print("Arquivo não encontrado")
        
    except ValueError as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()