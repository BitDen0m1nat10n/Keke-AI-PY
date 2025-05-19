from typing import List

file_name: str = "Keke_PY/tree_vs_linear__time_based__single_levels/experiment_logs/combined_logs.txt"

with open(file_name) as file:
    lines: List[str] = file.readlines()


solving_matrix: List[List[bool]] = []

for line in lines:
    if line.startswith("LEVEL AGENT EVALUATION:"):
        _, *results = line.split(':')
        solving_matrix.append([result.strip() != "----" for result in results])

solving_matrix = [solving_matrix[
    (2 * i) % len(solving_matrix) + (1 if 2 * i >= len(solving_matrix) else 0)
] for i in range(len(solving_matrix))]


for row in solving_matrix:
    print(*row)

import numpy as np
a = np.array(solving_matrix)
import matplotlib.pyplot as plt

plt.imshow(a)
plt.show()

print(np.sum(np.max(a, axis=0)))
print(a.shape)

levels_total: int = len(solving_matrix) // 2
linear_intentional_solved: int = sum(solving_matrix[i][i] for i in range(levels_total))
tree_intentional_solved: int = sum(solving_matrix[levels_total + i][i] for i in range(levels_total))
linear_total_solved: int = sum(any(solving_matrix[i][j] for i in range(levels_total)) for j in range(levels_total))
tree_total_solved: int = sum(any(solving_matrix[levels_total + i][j] for i in range(levels_total)) for j in range(levels_total))
print(f"linear intentional:\t{linear_intentional_solved}/{levels_total}={linear_intentional_solved / levels_total}")
print(f" trees intentional:\t{tree_intentional_solved}/{levels_total}={tree_intentional_solved / levels_total}")
print(f"      linear total:\t{linear_total_solved}/{levels_total}={linear_total_solved / levels_total}")
print(f"       trees total:\t{tree_total_solved}/{levels_total}={tree_total_solved / levels_total}")
