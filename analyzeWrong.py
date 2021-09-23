import numpy as np
import pickle

with open('wrong_indices.txt') as f:
    wrong_indices = pickle.load(f)
with open('testIndices.txt') as f:
    testIndices = pickle.load(f)

#print wrong_indices

mapToRealIndices = {i: testIndices[i] for i in range(len(testIndices))}
wrong_indices_reduced = [index/2 for index in wrong_indices if index % 2 == 0]


wrong_indices_mapped = []
for i in range(len(wrong_indices_reduced)):
    wrong_indices_mapped.append(mapToRealIndices[wrong_indices_reduced[i]])

print wrong_indices_mapped

pickle.dump(wrong_indices_mapped, open('wrong_indices_mapped.txt', 'wb'))