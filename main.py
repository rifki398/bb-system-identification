import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model.ball_beam import BallAndBeam, BallBeamNN

os.system("cls||clear")

## Create Ball and Beam Model
real_model = BallAndBeam()
id_model = BallBeamNN()

## Dataset Generation
np.random.seed(0)
N = 10000
X = []
Y = []

for _ in range(N):
    x = np.random.uniform(low=[-0.5, -1, -0.2, -2], high=[0.5, 1, 0.2, 2])  # [x, x_dot, theta, theta_dot]
    u = np.random.uniform(-1.0, 1.0)
    xdot = real_model.f(x,u)
    # xdot = ball_beam_dynamic(x, u)

    X.append(np.concatenate((x, [u])))
    Y.append(xdot)

X = np.array(X)
Y = np.array(Y)

# === 3. CONVERT TO TENSOR ===
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# === 4. DEFINE NEURAL NETWORK ===
optimizer = optim.Adam(id_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# === 5. TRAINING ===
epochs = 100
batch_size = 256
loss_history = []

for epoch in range(epochs):
    perm = torch.randperm(X_tensor.size(0))
    for i in range(0, X_tensor.size(0), batch_size):
        idx = perm[i:i+batch_size]
        x_batch = X_tensor[idx]
        y_batch = Y_tensor[idx]

        y_pred = id_model(x_batch) # same as model.forward(x_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    loss_history.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# === 6. PLOT LOSS ===
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.title("Training Loss")
plt.show()

# === 7. TEST ON UNSEEN INPUT ===
x_test = np.array([0.1, 0.2, 0.05, -0.1])
u_test = 0.3
x_input = torch.tensor(np.concatenate((x_test, [u_test])), dtype=torch.float32)

xdot_true = real_model.f(x_test, u_test)
xdot_pred = id_model(x_input).detach().numpy()

print("\nTrue xdot  :", xdot_true)
print("NN Predicted:", xdot_pred)
