import numpy as np

data=np.arange(0,10,1)

data=data.reshape(2,5)
print(data)
scale=np.array([1,2])
scale=scale.reshape(2,1)
print(data*scale)