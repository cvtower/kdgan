import scipy.io as sio

feature = sio.loadmat('feature.mat')
fc7 = feature['fc7']
print(type(fc7), fc7.shapec)