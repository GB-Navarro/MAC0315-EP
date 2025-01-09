import time
from scipy.optimize import linprog

# Definição de problemas de programação linear a serem resolvidos.
def define_problems():
    return {
        "Lista 1": {
            "Problema 1": {
                "type": "max",
                "c": [-3.00, -2.00, -1.00],
                "A_ub": [[2.00, 1.00, 1.00], [4.00, 2.00, 3.00], [1.00, 2.00, 3.00]],
                "b_ub": [14.00, 28.00, 10.00],
                "bounds": [(0, None), (0, None), (0, None)],
            },
            "Problema 2": {
                "type": "max",
                "c": [-5.00, -4.00, -3.00],
                "A_ub": [[3.00, 2.00, 1.00], [2.00, 3.00, 4.00], [1.00, 5.00, 3.00]],
                "b_ub": [15.00, 20.00, 18.00],
                "bounds": [(0, None), (0, None), (0, None)],
            },
            "Problema 3": {
                "type": "min",
                "c": [4.00, 6.00, 2.00],
                "A_ub": [[-1.00, -1.00, -1.00], [-2.00, -2.00, -3.00], [-3.00, -1.00, -2.00]],
                "b_ub": [-8.00, -12.00, -14.00],
                "bounds": [(0, None), (0, None), (0, None)],
            },
            "Problema 4": {
                "type": "max",
                "c": [-2.00, -3.00, -4.00],
                "A_ub": [[4.00, 2.00, 3.00], [3.00, 3.00, 2.00], [1.00, 4.00, 2.00]],
                "b_ub": [20.00, 15.00, 10.00],
                "bounds": [(0, None), (0, None), (0, None)],
            },
            "Problema 5": {
                "type": "min",
                "c": [3.00, 5.00, 7.00],
                "A_ub": [[-2.00, -3.00, -2.00], [-4.00, -1.00, -3.00], [-1.00, -2.00, -4.00]],
                "b_ub": [-12.00, -14.00, -10.00],
                "bounds": [(0, None), (0, None), (0, None)],
            },
        },
        "Lista 2": {
            "Problema 3": {
                "type": "max",
                "c": [-2.00, 1.00, -2.00],
                "A_eq": [[1.00, 1.00, 1.00], [1.00, 2.00, -4.00]],
                "b_eq": [12.00, -4.00],
                "bounds": [(1, 5), (1, 5), (1, 5)],
            },
            "Problema 4": {
                "type": "max",
                "c": [-1.00, 1.00, -1.00],
                "A_eq": [[1.00, 1.00, 1.00], [1.00, -3.00, 1.00]],
                "b_eq": [9.00, -3.00],
                "bounds": [(-1, 4), (-1, 4), (-1, 4)],
            },
        },
    }

# Função para resolver os problemas de programação linear definidos acima.
def solve_problem(problem):
    if "A_ub" in problem.keys():
        result = linprog(
            problem['c'],
            A_ub=problem['A_ub'],
            b_ub=problem['b_ub'],
            bounds=problem['bounds'],
            method='highs',
        )
    else:
        result = linprog(
            problem['c'],
            A_eq=problem['A_eq'],
            b_eq=problem['b_eq'],
            bounds=problem['bounds'],
            method='highs',
        )
    return result

# Função para exibir os resultados encontrados.
def display_result(problem_name, problem_type, result):
    print(f"{problem_name}:")
    if result.success:
        print(f"\nSolução encontrada: x1 = {result.x[0]:.2f}, x2 = {result.x[1]:.2f}, x3 = {result.x[2]:.2f}\n")
        z_value = result.fun * -1 if problem_type == 'max' else result.fun
        print(f"Valor {'máximo' if problem_type == 'max' else 'mínimo'} de Z = {z_value:.2f}\n")
    else:
        print("Não foi possível encontrar uma solução.\n")

def main():
    problems = define_problems()
    for list_name, problems_list in problems.items():
        print(f"------------------------------ {list_name} ------------------------------\n")
        for problem_name, problem_data in problems_list.items():
            start = time.time()
            result = solve_problem(problem_data)
            display_result(problem_name, problem_data['type'], result)
            elapsed = time.time() - start
            print(f"Tempo decorrido para resolver {problem_name}: {elapsed:.6f} segundos")

if __name__ == "__main__":
    main()
