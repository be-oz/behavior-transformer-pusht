import numpy as np

data = np.load("transformer_dataset.npy", allow_pickle=True)

print("Type:", type(data))
print("Shape:", getattr(data, 'shape', None))
print("First entry:", data[0])
