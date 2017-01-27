#!/Users/kennethyu/anaconda2/bin/python
import numpy as np
from PIL import Image
import os
import tempfile
import urllib2
import conda.cli

if __name__ == '__main__':
	print 'main is true'

# d=dict([['a',1]])
# print d
#
# file.readline()
# with open('testtxt.txt','r') as f:
# 	# l = [line.strip().split('  ') for line in f]
# 	# print l
# 	# l = dict(line.strip().split('\t') for line in f)
# 	l = f.readlines()
# 	print l[0].__class__
# 	print l

# f = urllib2.urlopen('http://baike.baidu.com/link?url=6aoSNI40wl1JOtGffqsNd2_GcODbEjvdpk5uGpcPUZPcDuJN_nvhHEjW6VBpzfZY')
# tfd,tfn = tempfile.mkstemp(suffix='tmp',prefix='cool',dir=os.path.curdir)
# with open(tfn,'wrb') as tf:
# 	tf.write(f.read())
# with open (tfn, 'rb') as tf:
# 	print tf.readlines()
# 	tf.close()

# fd, fname = tempfile.mkstemp()
# print fd
# print fname

# from matplotlib import cm
#
# print hasattr(cm, 'gray_r')


# print hasattr(img_obj,'__array_interface__')
#
# # print  np.array(img_obj.__getattr__('__array_interface__'))
# d= img_obj.__getattr__('__array_interface__')
# print hasattr(d, '__array_interface__')
#
#
#
# y = list([[1,2],[2,3]])
# print hasattr(y,'__array_interface__')
# print y.__class__
# z = np.array(y)
# print z

