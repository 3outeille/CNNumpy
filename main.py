def f():
    """

    """
    #INITIALIZATION

    #Initialize parameters.
    F1, b1 = initializeFilter(n_f = 6, f = 5, n_C = 1)
    F2, b2 = initializeFilter(n_f = 16, f = 5, n_C = 6)
    W3, b3 = initializeParamFc(120, 400)
    W4, b4 = initializeParamFc(84, 120)
    W5, b5 = initializeParamFc(10, 84)
    
    cache = {} #Use to keep track of derivatives.

    #--------------------
    #FORWARD PROPAGATION.
    #--------------------

    #1st layer Convolution.
    conv1 = convFwd(X, F1, b1, p = 0, s = 1) #(28 x 28 x 6).
    conv1_act = actTanh(conv1) 
    pool1 = avgpoolFwd(conv1_act, f = 2, p = 0, s = 2) #(14 x 14 x 6).

    #2nd layer Convolution.
    conv2 = convFwd(pool1, F2, b2, p = 0, s = 1) #(10 x 10 x 6).
    conv2_act = actTanh(conv2)
    pool2 = avgpoolFwd(conv2_act, f = 2, p = 0, s = 2) #(5 x 5 x 16).

    #Flatten.
    pool2_flat = pool2.flatten() #(400, 1).

    #1st Fully connected layer.
    fc1 = fcFwd(pool2_flat, W3, b3) #(120 x 1) 
    fc1_act = actTanh(fc1)

    #2nd Fully connected layer.
    fc2 = fcFwd(fc1_act, W4, b4) #(84 x 1)
    fc2_act = actTanh(fc2)

    #Output layer.
    yHat = fcFwd(fc2_act, W5, b5) #(10 x 1)

    #Compute cost.
    cost = costFunction(yHat, y)

    #--------------------
    #BACKWARD PROPAGATION.
    #--------------------

    #Error at yHat.
    deltaL = yHat - y #(10, 1).
    #Backpropagate error in weight/bias between yHat and fc2.
    dW5, db5 = fcBack(deltaL, fc2_act, W5, b5) #(10, 84).
    
    #Error at fc2.
    deltaL = np.dot(W5.T, deltaL) * actTanhBack(fc2) #(84, 1).
    #Backpropagate error in weight/bias between fc2 and fc1.
    dW4, db4 = fcBack(deltaL, fc1_act), W4, b4) #(84, 120).
    
    #Error at fc1.
    deltaL = np.dot(W4.T, deltaL) * actTanhBack(fc1) #(120, 1).
    #Backpropagate error in weight/bias between fc1 and pool2.
    dW3, db3 = fcBack(deltaL, pool2_flat, W3, b3) #(120, 400).

    #Error at pool2_flat.
    deltaL = np.dot(W3.T, deltaL) * actTanhBack(pool2_flat) #(400, 1).
    #Reshape pool2_flat into pool2.  
    pool2 = np.reshape(deltaL, pool2.shape) #(5 x 5 x 16).

    #Backpropagate error in pool2 to conv2_act.
    conv2_act = avgpoolBack(conv2_act, pool2, f = 2, p = 0, s = 2) #(10 x 10 x 16).
    #Backpropagate through derivative of tanH.
    conv2 = actTanhBack(conv2_act)
    #Backpropagate error in filter/bias between conv2 and pool1.
    dF2, db2 = convBack(conv2, F2, b2) #(5 x 5 x 6) and (1 x 1 x 6)
        
    #Error at pool1.

    #Backpropagate error in pool1 to conv1.
    #Backpropagate through derivative of tanH.
    #Backpropage error in filter/bias between conv1 and X.


    #Add dW and db to gradient

    #------
    #UPDATE
    #------

    #Update parameters.
    





