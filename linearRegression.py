import matplotlib.pyplot as plt
import numpy as np


x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
# A graph of x y values, going to use linear regression for a model of data points (given x, predict y)

# Get best fit line
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.show()

# Linear regression draws a line of best fit with equal or as close to equal data points on the left as on the right
# of the line

# In 3D, given two data points, you can predict the third
