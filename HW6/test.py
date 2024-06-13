import numpy as np

print(np.hypot(3, 5))
print(np.sqrt(3**2 + 5**2))

x = np.array([-1, +1, +1, -1])

y = np.array([-1, -1, +1, +1])

print(np.arctan2(y, x) * 180 / np.pi)
print(np.arctan2(x, y) * 180 / np.pi)
