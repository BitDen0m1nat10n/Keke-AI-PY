from typing import List

file_name: str = [
    "Keke_PY/experiment_logs/single_level_tree_optimization/single_level_optimization_combined_logs.txt",
    "Keke_PY/experiment_logs/single_level_tree_optimization/long_100_gens_single_level_optimization_combined_logs.txt",
    "Keke_PY/tree_vs_linear__time_based__single_levels/experiment_logs/combined_logs.txt"
][2]

with open(file_name) as file:
    lines: List[str] = file.readlines()


solving_matrix: List[List[bool]] = []

for line in lines:
    if line.startswith("LEVEL AGENT EVALUATION:"):
        _, *results = line.split(':')
        solving_matrix.append([result.strip() != "----" for result in results])


for row in solving_matrix:
    print(*row)

import numpy as np
a = np.array(solving_matrix)
import matplotlib.pyplot as plt

plt.imshow(a)
plt.show()

print(np.sum(np.max(a, axis=0)))
print(a.shape)
print(148/184)