import numpy as np
import numpy.lib.stride_tricks as st
import matplotlib.pyplot as plt


d={'1':1}
iter(d)

a = np.arange(24).reshape((2,4,3))
print '==a :', a


b = np.transpose(a,(0,2,1))

b.ravel()
print '@@ b: ',b

c=b.reshape((6,4))
print '====c:', c


# print np.max(d), np.unravel_index( np.argmax(d),(2,2))






