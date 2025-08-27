import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train(X,Y,model):
    '''
    **Input** ------------------------------------- \n
    `X`: input data set for NN \n
    `Y`: output data set for NN \n
    `model`: identified model (using NN class) \n
    **Output** ----------------------------------- \n
    `loss_history`: Loss function value \n
    `model`: trained model
    '''
    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    # Define the Neural Network
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training
    epochs = 100
    batch_size = 256
    loss_history = []

    for epoch in range(epochs):
        perm = torch.randperm(X_tensor.size(0))
        for i in range(0, X_tensor.size(0), batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X_tensor[idx]
            y_batch = Y_tensor[idx]

            y_pred = model(x_batch) # same as model.forward(x_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return loss_history, model