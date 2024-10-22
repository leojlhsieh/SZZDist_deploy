# %%
import torch
import torch.nn as nn
import torch.optim as optim


def get_gradients(model):
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:  # Check if gradients exist
            gradients[name] = param.grad.data.clone()  # Clone to avoid modifications
            print(f"Gradient for {name}: collected {param.grad.data}")

        else:
            print(f"Gradient for {name}: is None")
    return gradients

# Step 1: Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Step 2: Initialize the model and optimizer
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for i in range(2):
    # Step 3: Create some dummy input and target data
    input_data = torch.randn(5, 10)  # 5 samples, 10 features
    target_data = torch.randn(5, 1)   # 5 target values

    # Step 4: Forward pass
    output = model(input_data)
    loss = nn.MSELoss()(output, target_data)

    # Step 5: Backward pass
    optimizer.zero_grad()  # Zero the gradients before backward pass


    loss.backward()        # Compute gradients

    # Step 7: (Optional) Perform an optimization step
    optimizer.step()
    gradient = get_gradients(model)


# Step 6: Access and record the gradients
gradients = {}
for name, param in model.named_parameters():
    if param.grad is not None:  # Check if gradients exist
        gradients[name] = param.grad.data.clone()  # Clone to avoid modifications
        print(f"Gradient for {name}: {param.grad.data}")

# %%
param.data.histogram(bins=64)  # not implemented for CUDA tensors

# %%
# Get the learning rate of the optimizer
for param_group in optimizer.param_groups:
    learning_rate = param_group['lr']
    print(f"Learning rate: {learning_rate}")


# %%
# Get parameter information of model
for name, param in model.named_parameters():
    print(f"Parameter name: {name}")
    print(f"Parameter shape: {param.shape}")
    print(f"Parameter requires_grad: {param.requires_grad}")
    print(f"Parameter data: {param.data}")
    # historgam of the parameter data
    hist = param.data.histc(bins=10, min=param.data.min().item(), max=param.data.max().item())
    print("-" * 50)

# %%
# get the historgam of the parameter data, and plot it
import matplotlib.pyplot as plt

for name, param in model.named_parameters():
    hist = param.data.histc(bins=10, min=param.data.min().item(), max=param.data.max().item())
    plt.figure()
    plt.bar(range(10), hist.numpy())
    plt.title(f"Histogram of {name}")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.show()


# %%
# count number of trainable parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")