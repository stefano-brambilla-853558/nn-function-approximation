import streamlit as st
import torch
import plotly.graph_objs as go
from sympy import sympify, symbols

# Header
st.title("Universal Approximation Theorem")
st.markdown("A simple demo for its application to neural networks")
st.header("")

#############################
# Define the neural network
#############################

n_layers = st.sidebar.slider(label = "Number of hidden layers", 
                        min_value = 1,
                        max_value= 10,
                        value = 2,
                        step = 1)

h_sizes = [1] + [st.sidebar.radio(label = f"Size of layer {i + 1}",  
                        options = [64, 128, 256],
                        index = 0,
                        horizontal = True)          
            for i in range(n_layers)]

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(torch.nn.Linear(h_sizes[k], h_sizes[k+1]))
        self.output = torch.nn.Linear(h_sizes[-1], 1)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.output(x)

net = Net()    
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

#############################
# Define the function to approximate
#############################
col1, col2 = st.columns(2)
functions = ["sin(x)", "x^3", "cos(3*x)"]
choice = col1.radio(label="Select a pre-defined function", 
                  options=functions,
                  index=0,
                  horizontal=True)
expression = col2.text_input(label= "or choose you own", value=choice)
x = symbols('x')
function = sympify(expression)

def f(t):
    tens = torch.clone(t)
    tens.apply_(lambda val: (float(function.subs(x, val))))
    return tens

#############################
# Define the intervals
#############################
st.write("")
col1, col2 = st.columns(2)
col1.write("Training interval:")
col1A, col1B = col1.columns(2)
train_min = col1A.number_input(label="min", value=-1., key=1)
train_max = col1B.number_input(label="max", value=1., min_value=train_min+0.01, max_value=100., step=1., key=2)
col2.write("Test interval:")
col2A, col2B = col2.columns(2)
test_min = col2A.number_input(label="min", value=train_min, key=3)
test_max = col2B.number_input(label="max", value=train_max, min_value=test_min+0.01, max_value=100., step=1., key=4)

x_train = torch.linspace(train_min,train_max, 20)
x_test = torch.linspace(test_min,test_max, 20)


#############################
# Train
#############################
for epoch in range(1000):
    running_loss = 0.0
    optimizer.zero_grad()
    outputs = net(x_train.unsqueeze(1))
    loss = criterion(outputs.squeeze(), f(x_train))
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    
    if epoch % 100 == 0:
        print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))


#############################
# Plot
#############################
actual_y_train = torch.tensor([f(p) for p in x_train])
actual_y_test = torch.tensor([f(p) for p in x_test])
predicted_y = net(x_test.unsqueeze(1)).squeeze()

fig  = go.Figure()
fig.add_trace(go.Scatter(x=x_train, 
                         y=actual_y_train, 
                         mode='lines', 
                         name='Actual Function (train set)',
                         line=dict(color='#8ECEFF')))
fig.add_trace(go.Scatter(x=x_test, 
                         y=actual_y_test, 
                         mode='lines', 
                         name='Actual Function (test set)',
                         line=dict(color='#8ECEFF', dash='dash')))
fig.add_trace(go.Scatter(x=x_test, 
                         y=predicted_y.detach().numpy(), 
                         mode='markers', 
                         name='Predicted Function',
                         line=dict(color='#0068C9')))

st.plotly_chart(fig)

