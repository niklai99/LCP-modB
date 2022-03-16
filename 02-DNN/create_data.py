import numpy as np

def sample_data(N=3000, B=100):

    x = (np.random.random((N,2))-0.5)*B

    return x


def labeling_function(x, c=1):
    if c==1:
        return ((x[:,0]>-20) & (x[:,1]>-40) & (x[:,0]+x[:,1] < 40)).astype(int)
    elif c==2:
        return ((np.sign(x.sum(axis=1))*np.sign(x[:,0]))
                *np.cos(np.linalg.norm(x, axis=1)/(2*np.pi))>0).astype(int)


def assign_label(x):
    y = labeling_function(x, c=1)
    return y


def create_data(N=4000, B=100, c='triang', dim=3):
    if c=='triang':
        x = (np.random.random((N,2))-0.5)*B
        y = ((x[:,0]>-20) & (x[:,1]>-40) & (x[:,0]+x[:,1] < 40)).astype(int)
    elif c=='weird':
        x = (np.random.random((N,2))-0.5)*B
        y = ((np.sign(x.sum(axis=1))*np.sign(x[:,0]))
             *np.cos(np.linalg.norm(x, axis=1)/(2*np.pi))>0).astype(int)
    elif c=='rad':
        r1, r2 = B/10, B/10 +B/5
        choice = np.random.random(N)
        r = (np.array([r1 if choice[i] < 0.5 else r2 for i in range(N)])
            + (np.random.random(N)-0.5)*B/5)
        alpha = 2*np.pi*np.random.random(N)
        x = np.empty((N,2))
        x[:,0] = np.cos(alpha)*r
        x[:,1] = np.sin(alpha)*r
        y = (choice < 0.5).astype(int)
    elif c=='radNd':
        r1, r2 = B/10, B/10 +B/5
        choice = np.random.random(N)
        r = (np.array([r1 if choice[i] < 0.5 else r2 for i in range(N)])
            + (np.random.random(N)-0.5)*B/5)
        thetas = np.concatenate((np.pi*np.random.random((N, dim-2)), 
                                 2*np.pi*np.random.random((N,1))), axis=1)
        #print(thetas[0:100,:])
        x = np.empty((N,dim))
        x[:,0] = r*np.cos(thetas[:,0])
        for i in range(1,dim-1):
            x[:,i] = r*np.cos(thetas[:,i])
            for j in range(i):
                x[:,i] = x[:,i]*np.sin(thetas[:,j])
        x[:,dim-1] = r*np.sin(thetas[:, dim-2])
        for i in range(dim-2):
            x[:,dim-1] = x[:,dim-1]*np.sin(thetas[:, i])
        y = (choice < 0.5).astype(int)

    return (x, y)


def create_grid(B, dx):

    xgrid = np.arange(-0.5*B,0.5*B+dx, dx)
    l_grid = xgrid.shape[0]
    grid = np.zeros((l_grid*l_grid, 2))

    k=0
    for i in range(l_grid):
        for j in range(l_grid):
            grid[k,:] = (xgrid[j], xgrid[i])
            k=k+1
            
    return grid

