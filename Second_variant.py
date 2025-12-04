import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os

OUTPUT_DIR = "results_LWR"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(OUTPUT_DIR)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LAYERS = [2, 64, 64, 64, 64, 1]
EPOCHS = 5000
LR = 1e-3


class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = nn.Tanh()

        layer_list = []
        for i in range(self.depth):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
        self.layers = nn.ModuleList(layer_list)

        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        a = x
        for i in range(self.depth - 1):
            a = self.activation(self.layers[i](a))
        a = self.layers[-1](a)
        return a


def physics_loss(model, col_points):
    col_points.requires_grad = True
    rho = model(col_points)

    grads = torch.autograd.grad(rho, col_points, torch.ones_like(rho), create_graph=True)[0]

    rho_t = grads[:, 0:1]  # похідна по t
    rho_x = grads[:, 1:2]  # похідна по x

    residual = rho_t + (1.0 - 2.0 * rho) * rho_x

    loss_f = torch.mean(residual ** 2)
    return loss_f


def get_data(n_collocation, n_boundary, n_initial):
    t_col = np.random.uniform(0, 1, (n_collocation, 1))
    x_col = np.random.uniform(0, 1, (n_collocation, 1))
    col_data = np.hstack((t_col, x_col))

    x_ic = np.random.uniform(0, 1, (n_initial, 1))
    t_ic = np.zeros((n_initial, 1))

    rho_ic_target = np.ones((n_initial, 1)) * 0.2
    mask_dense = (x_ic < 0.5)
    rho_ic_target[mask_dense] = 0.8

    rho_ic_target += np.random.normal(0, 0.005, rho_ic_target.shape)

    ic_input = np.hstack((t_ic, x_ic))

    t_bc = np.random.uniform(0, 1, (n_boundary, 1))
    x_bc_left = np.zeros_like(t_bc)

    rho_bc_left_target = 0.2 + (0.8 - 0.2) * t_bc

    bc_input_left = np.hstack((t_bc, x_bc_left))

    return (torch.tensor(col_data, dtype=torch.float32, device=device),
            torch.tensor(ic_input, dtype=torch.float32, device=device),
            torch.tensor(rho_ic_target, dtype=torch.float32, device=device),
            torch.tensor(bc_input_left, dtype=torch.float32, device=device),
            torch.tensor(rho_bc_left_target, dtype=torch.float32, device=device))


model = PINN(LAYERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

col_pts, ic_in, ic_target, bc_l_in, bc_l_target = get_data(8000, 2000, 2000)

print("Starting training (LWR Traffic Model)...")
start_time = time.time()
loss_history = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    loss_pde = physics_loss(model, col_pts)

    ic_pred = model(ic_in)
    loss_ic = torch.mean((ic_pred - ic_target) ** 2)

    bc_l_pred = model(bc_l_in)
    loss_bc = torch.mean((bc_l_pred - bc_l_target) ** 2)

    loss = loss_pde + 10 * loss_ic + 10 * loss_bc

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

print(f"Training finished in {time.time() - start_time:.1f}s")

plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Total Loss')
plt.yscale('log')
plt.title("LWR Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_history.png"), dpi=300)
plt.close()

def save_heatmap():
    x = np.linspace(0, 1, 200)
    t = np.linspace(0, 1, 200)
    X, T = np.meshgrid(x, t)  # Сітка

    input_grid = np.stack([T.flatten(), X.flatten()], axis=1)
    input_tensor = torch.tensor(input_grid, dtype=torch.float32, device=device)

    with torch.no_grad():
        rho_pred = model(input_tensor).cpu().numpy().reshape(200, 200)

    plt.figure(figsize=(8, 6))
    plt.imshow(rho_pred, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label='Density (rho)')
    plt.xlabel('Position (x)')
    plt.ylabel('Time (t)')
    plt.title('LWR Traffic Flow Density (Prediction)')

    filename = os.path.join(OUTPUT_DIR, "lwr_heatmap.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved heatmap to {filename}")


def save_snapshot(t_val):
    x = np.linspace(0, 1, 200)
    t = np.ones_like(x) * t_val

    input_data = np.stack([t, x], axis=1)
    input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device)

    with torch.no_grad():
        rho_pred = model(input_tensor).cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(x, rho_pred, 'b-', linewidth=2, label=f'PINN t={t_val}')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title(f'Traffic Density Profile at t={t_val}')
    plt.grid(True)
    plt.legend()

    filename = os.path.join(OUTPUT_DIR, f"profile_t_{t_val}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved profile to {filename}")


save_heatmap()
save_snapshot(0.0)
save_snapshot(0.2)
save_snapshot(0.5)
save_snapshot(0.8)

print("Done! Check the 'results_LWR' folder.")