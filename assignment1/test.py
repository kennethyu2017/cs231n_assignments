import numpy as np

import scipy as sp
import matplotlib.pyplot as plt
np.random.seed(10)
x = np.random.rand(10,1)

np.random.seed(10)
y= np.random.rand(10,1)

np.random.seed(10)
z = np.random.rand(10,1)

print (x==y)
print  y ==z

# D = [1e2,1e3,1e5, 1e6]
#
#
# for d in D:
#     x = np.zeros((1000,1))
#     for i in xrange(np.int64(d)):
#         x += np.random.randn(1000,1)
#     print 'd: %d, var: %f, max z: %f, min z: %f, mean: %d' % (d, np.var(x),np.max(x), np.min(x),np.mean(x))
#
#
#

    #
    # print np.std(x), np.var(x), np.max(x), np.min(x)
    # z = []
    # for i in xrange(1000):      #1000 random data
    #     y = np.random.randn(d,1)  #data
    #     z.append(np.dot(np.transpose(x),y))
    #
    # print 'd: %d, var: %f, max z: %f, min z: %f, mean: %d' % (d, np.var(z),np.max(z), np.min(z),np.mean(z))

#     plt.figure()
#     plt.subplot(1,len(D), D.index(d)+1)
#     plt.scatter(xrange(1000), z)
#
# plt.show()


# x   =  np.arange(16,dtype=np.int64).reshape(4,4)
#
# #
# # print x
# #
# # print np.insert(x,2,[-34,-54,-34524345,-34534],axis=-1)
# # #print np.tile(x,(2,3))
# # #print np.repeat(x,2, axis=1)
#
# x = np.diag([1,2,3,4])
# print x.transpose()
# print np.flipud(x)
# print np.rot90(x)


#
# a = np.array([1, 2, 3])
# b = np.array([2, 3, 4])
# print np.stack((a, b)).shape
#
# # array([[1, 2, 3],
# #        [2, 3, 4]])
#
#
#
# print np.concatenate((a, b), axis=1)
#
# print np.dstack((a,b))
#
# # array([[1, 2],
#        [2, 3],
#        [3, 4]])




# def test(*ary):
#     print np.atleast_3d(ary)
#
# a1 = np.array([34,45,45])
# a2 = np.array([123,3,3,3])
#
# print np.expand_dims(a1,axis=3)

#test(a1,a2)




#
# dt = np.float64
#
# #x = np.random.randn(3,100)
#
# x = np.array([[[1,2,45,4],[3,4,345,45],[5,6,345,2345]],[[456,3452,5670,3452],
#                                                         [4563,3,34,234],[345,34,2345,345345]]])
# # y = np.array([2,3,4,5,6])
# print x.shape
# print x
# y = np.rollaxis(x,0,start=3)
# print y.shape
# print y
#
#
# #y = np.tanh(x.ravel())
# #print np.gradient(x)
# # print np.convolve(x,y,mode='full')
# # print y
# # np.where
# #z = [i for i in y.item()]
# #print z.reverse()
#
# #plt.figure(111)
# #plt.scatter(x,y)
# #plt.show()


"""

x = []
y = []
z = []

for i in xrange(1000):
    x.append(i)
    y.append(10**np.random.uniform(3,5))
    z.append(np.random.uniform(10**3, 10**5))


plt.figure()
plt.subplot(131)
#plt.hist(sorted(y),10)
plt.scatter(x, y)

plt.subplot(132)
#plt.hist(sorted(z),10)
plt.scatter(x,z)

plt.subplot(133)
plt.scatter(x,np.log10(y))

plt.show()

"""