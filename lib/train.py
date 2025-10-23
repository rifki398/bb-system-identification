import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def train(nn_model,real_model,N):
    '''
    **Input** ------------------------------------- \n
    `X`: input data set for NN \n
    `Y`: output data set for NN \n
    `model`: identified model (using NN class) \n
    **Output** ----------------------------------- \n
    `loss_history`: Loss function value \n
    `model`: trained model
    '''
    X,Y  = generate_dataset(real_model,N,False)
    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    # Define the Neural Network
    optimizer = optim.Adam(nn_model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Training
    epochs = 110
    batch_size = 256
    loss_history = []

    for epoch in range(epochs):
        perm = torch.randperm(X_tensor.size(0))
        for i in range(0, X_tensor.size(0), batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X_tensor[idx]
            y_batch = Y_tensor[idx]

            y_pred = nn_model(x_batch) # same as model.forward(x_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return loss_history, nn_model

def generate_dataset(model, N=10000, normalize=True):
    '''
    **Input** -------------------- \n
    `N`: data set size \n
    `model`: Real plant model \n
    **Output** ------------------- \n
    `X`: input data set for NN \n
    `Y`: output data set for NN
    '''
    np.random.seed(0)
    X, Y = [], []

    for _ in range(N):
        x = np.random.uniform(low=[-0.6, -0.3, -np.deg2rad(10), -np.deg2rad(40)], 
                              high=[0.6, 0.3, np.deg2rad(10), np.deg2rad(40)])
        u = np.random.uniform(-1.6, 1.6)
        xdot = model.f(x, u)
        X.append(np.concatenate((x, [u])))
        Y.append(xdot)
    
    X = np.array(X)
    Y = np.array(Y)

    if normalize:
        # Meand and std
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0)
        
        # Avoid dividing by 0
        X_std[X_std == 0] = 1e-8
        Y_std[Y_std == 0] = 1e-8

        # Normalize
        X_norm = (X - X_mean) / X_std
        Y_norm = (Y - Y_mean) / Y_std

        return X_norm, Y_norm
    
    else:
        return X, Y