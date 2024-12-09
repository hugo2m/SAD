import numpy as np # type: ignore

# Função para calcular as prioridades e verificar consistência
def ahp(matrix):
    """
    Calcula o vetor de prioridades e verifica a consistência da matriz de comparações pareadas.
    """
    eig_values, eig_vectors = np.linalg.eig(matrix)
    max_eigenvalue = np.max(eig_values)
    eig_vector = eig_vectors[:, np.argmax(eig_values)].real
    priorities = eig_vector / eig_vector.sum()

    # Consistency Index (CI)
    n = matrix.shape[0]
    ci = (max_eigenvalue - n) / (n - 1)

    # Random Index (RI) para diferentes valores de n
    ri_values = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    ri = ri_values.get(n, 1.49)  # Para n > 9, RI é aproximadamente 1.49

    # Consistency Ratio (CR)
    cr = ci / ri if ri != 0 else 0

    return priorities, cr

# Definindo a matriz de comparações pareadas (critérios)
criteria_matrix = np.array([
    [1,   3,   1/5, 7,   5],  # Custo
    [1/3, 1,   1/7, 5,   3],  # Facilidade de uso
    [5,   7,   1,   9,   8],  # Funcionalidades
    [1/7, 1/5, 1/9, 1,   1/3],# Suporte técnico
    [1/5, 1/3, 1/8, 3,   1],  # Compatibilidade
])

# Calculando as prioridades e a consistência dos critérios
criteria_priorities, cr_criteria = ahp(criteria_matrix)

# Definindo as matrizes de comparações para as alternativas em relação a cada critério
# Custo
alt_cost_matrix = np.array([
    [1,   3,   1/5],  # Software A
    [1/3, 1,   1/7],  # Software B
    [5,   7,   1],    # Software C
])

# Facilidade de uso
alt_ease_matrix = np.array([
    [1,   1/2, 3],  # Software A
    [2,   1,   5],  # Software B
    [1/3, 1/5, 1],  # Software C
])

# Funcionalidades
alt_features_matrix = np.array([
    [1,   1/4, 5],  # Software A
    [4,   1,   7],  # Software B
    [1/5, 1/7, 1],  # Software C
])

# Suporte técnico
alt_support_matrix = np.array([
    [1,   3,   1/3],  # Software A
    [1/3, 1,   1/5],  # Software B
    [3,   5,   1],    # Software C
])

# Compatibilidade
alt_compatibility_matrix = np.array([
    [1,   5,   1/7],  # Software A
    [1/5, 1,   1/9],  # Software B
    [7,   9,   1],    # Software C
])

# Calculando as prioridades para cada matriz de alternativas
alt_cost_priorities, _ = ahp(alt_cost_matrix)
alt_ease_priorities, _ = ahp(alt_ease_matrix)
alt_features_priorities, _ = ahp(alt_features_matrix)
alt_support_priorities, _ = ahp(alt_support_matrix)
alt_compatibility_priorities, _ = ahp(alt_compatibility_matrix)

# Pesos globais dos critérios (baseados nos resultados anteriores)
criteria_weights = criteria_priorities

# Combinação linear para calcular o peso global das alternativas
final_scores = (
    criteria_weights[0] * alt_cost_priorities +
    criteria_weights[1] * alt_ease_priorities +
    criteria_weights[2] * alt_features_priorities +
    criteria_weights[3] * alt_support_priorities +
    criteria_weights[4] * alt_compatibility_priorities
)

# Exibindo os resultados
print("Pesos globais dos critérios:", criteria_priorities)
print("Razão de consistência dos critérios:", cr_criteria)
print("\nPesos finais das alternativas:")
print("Software A:", final_scores[0])
print("Software B:", final_scores[1])
print("Software C:", final_scores[2])
