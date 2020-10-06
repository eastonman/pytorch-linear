from torch import nn
import torch
from torch import Tensor


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(3, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2e-4)

with open("input.txt", 'r') as input:
    total = int(input.readline())
    x_raw = input.readline().split()
    y_raw = input.readline().split()
    x_float = [[float(x)] for x in x_raw[0:total]]
    y_float = [[float(y)] for y in y_raw[0:total]]

x_data = Tensor(x_float)
x_data = torch.cat(([x_data**i for i in range(1, 4)]), 1)
y_data = Tensor(y_float)

del x_raw, y_raw, x_float, y_float


# Training loop
for epoch in range(20000):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# After training
hour_var = Tensor([[10.0, 100.0, 1000.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  10, model(hour_var).data[0][0].item())
