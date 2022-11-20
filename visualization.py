"""This module is for visualization of graphs."""
import matplotlib.pyplot as plt
import math

fig, ax = plt.subplots()

x = [i*(1/100) for i in range(-100, 100)]
y1 = [x*x for x in x]
y2 = [math.sqrt(1-x*x) for x in x]
ax.axis('equal')
ax.plot(x, y2, label='y2')
ax.plot(x, y1, label='y1')
ax.plot(x, x, label='x')
ax.set_title('title')
ax.set_ylabel('ylabel')
ax.set_xlabel('xlabel')
ax.legend()
# plt.show()
