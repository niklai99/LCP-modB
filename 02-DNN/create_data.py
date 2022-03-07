import numpy as np

def sample_data(N=3000, B=100):

    x = (np.random.random((N,2))-0.5)*B

    return x


def labeling_function(x,c=1):
    r=0
    if c==1:
        if x[0]>-20 and x[1]>-40 and x[0]+x[1] < 40:
            r=1
    if c==2:
        if (np.sign(x.sum())*np.sign(x[0]))*np.cos(np.linalg.norm(x)/(2*np.pi))>0:
            r=1
    return r

def assign_label(x):
    N = x.shape[0]
    y = np.zeros(N)
    for i in range(N):
        y[i] = labeling_function(x[i], c=1)
    return y










