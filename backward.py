def fcBack(deltaL, A_prev, W, b):
    """
        Returns gradients computation of W and b.

        Parameters:
        -'deltaL': error at previous layer.
        -'A_prev': previous layer.
        -'W': weight between A_prev/current layer. 
        -'b': bias between A_prev/current layer.

        Returns:
        -'dW': gradient of weight.
        -'db': gradient of bias.
    """
    m = A_prev.shape(0)
    dW = (1/m) * np.dot(deltaL, A_prev.T)
    db = (1/m) * np.sum(deltaL, axis=1) #If problem, check here.

    return dW, db

def avgpoolBack(A_conv_act, A_pool, f, p , s):
    """ 
        Distributes the error to the activate convolution layer.
        
        Parameters:
        -'A_conv_act': activate convolution layer.
        -'A_pool': pooling layer.
        -'deltaL': error of next layer.
        -'f': filter size.
        -'p': input padding.
        -'s': input stride.

        Returns:
        -'A_conv_act_back': Activate convolution layer updated with error.
    """
    m, n_H, n_W, n_C = A_pool.shape  
    A_conv_act_back = np.copy(A_conv_act)        

    for i in range(m):

        for h in range(n_H): 
            h_start = h * s
            h_end = h_start + f

            for w in range(n_W):
                w_start = w * s
                w_end = w_start + f
                
                for c in range(n_C):
                    #Compute average for a given value.
                    average = A_pool[i, h, w, c] / (n_H * n_W)
                    #Create a filter of size (f, f) filled with average.
                    filter_average = np.full((f, f) , average)

                    A_conv_act_back[i, h_start:h_end, w_start:w_end, c] += filter_average 

    return A_conv_act_back

def actTanhBack(A_conv_act):
     """
        Apply derivative of activation function.
        
        Parameters:
        -'A_conv_act': output of avgpoolBack() function.
        
        Returns:
        -'A_conv': tanh function applied on conv output.

    """
    A_conv = dTanh(A_conv)

    return A_conv

def convBack(A_conv, A_pool, F, b, p, s):
    """


    """
    m, n_H, n_W, n_C = A_conv.shape  
    n_f, f, f, _ = F.shape
    
    for i in range(m):

        for h in range(n_H): 
            h_start = h * s
            h_end = h_start + f

            for w in range(n_W):
                w_start = w * s
                w_end = w_start + f
                
                    F[] += A_conv[i, h_start:h_end, w_start:w_end] * A_pool[]
