def splitting(x, y, verbose=False):

    N = len(x)
    
    train_test_frac  = 0.7
    valid_train_frac = 0.3

    N_train = int(N*train_test_frac*(1-valid_train_frac))
    N_valid = int(N*train_test_frac*valid_train_frac)
    N_test  = int(N*(1-train_test_frac))

    train_idx = N_train
    valid_idx = N_train+N_valid
    test_idx  = N_train+N_test

    x_train    = x[0:train_idx]
    y_train    = y[0:train_idx]
    x_val      = x[train_idx:valid_idx]
    y_val      = y[train_idx:valid_idx]
    x_test     = x[valid_idx:test_idx]
    y_test     = y[valid_idx:test_idx]

    if verbose:
        print('N_train =',N_train,'  N_val =',N_valid,'  N_test =',N_test,'  L =',L,'  n_class =',n_class)

    return x_train, y_train, x_val, y_val, x_test,  y_test