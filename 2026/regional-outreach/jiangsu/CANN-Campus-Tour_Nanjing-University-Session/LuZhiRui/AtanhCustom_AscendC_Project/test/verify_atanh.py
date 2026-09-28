import numpy as np

x = np.random.uniform(
    -0.9,0.9,
    (4,8,16,16)
).astype(np.float16)

gold = np.arctanh(x.astype(np.float32)).astype(np.float16)

print("input:", x.shape)
print("gold generated")
