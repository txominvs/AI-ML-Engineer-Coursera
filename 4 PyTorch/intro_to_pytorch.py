import torch

#
# 1D tensors
#
tensor_1D = torch.tensor([1,2,3,4,5], dtype=torch.int32)
tensor_1D = torch.FloatTensor([1,2,3,4,5])
tensor_1D[3]; tensor_1D[2:5] # element access
tensor_1D[9:12] = torch.tensor([9, 8, 7]) # assign to slice
multiple_indexes = [10, 14]; tensor_1D[multiple_indexes] = 1000.
tensor_1D.dtype # torch.int64 torch.float32
tensor_1D.type() # torch.LongTensor torch.FloatTensor
tensor_1D = tensor_1D.type(torch.FloatTensor) # change type
tensor_1D.size() # shape of tensor
tensor_1D.ndimension() # rank of tensor
reshaped_tensor = tensor_1D.view(rows, cols, ...) # use rows=-1 to infer the value automatically

pytorch_tensor = torch.from_numpy(numpy_tensor) # BE CAREFUL! variable POINTS TO REFERENCE of numpy_tensor, changes in both will be synced!
pytorch_tensor = torch.from_numpy(pandas.Series([0,1,2.2]) .values)
numpy_tensor = pytorch_tensor.numpy()
python_array = pytorch_tensor.tolist()
python_number = pytorch_tensor[indexes, ...].item() # convert 0D tensor to number

tensor_operations = 3.14*pytorch_tensor -9*pytorch_tensor
broadcast_sum_to_each_entry = pytorch_tensor +  2.7271
elementwise_multiplication = pytorch_tensor * pytorch_tensor
dot_product = torch.dot(pytorch_tensor, pytorch_tensor)
pytorch_tensor.mean(); pytorch_tensor.std(); pytorch_tensor.max();
elementwise = torch.sin(pytorch_tensor)

torch.linspace(min_value, max_value, steps=5)

plt.plot(x=pytorch_tensor.numpy(), y=pytorch_tensor.numpy())


#
# 2D tensors
#
tensor_2D = torch.tensor([
    [11, 12, 13],
    [21, 22, 23],
    [31, 32, 33]
])
tensor_2D.ndimension() # = rank = 2
tensor_2D.shape == tensor_1D.size() # (number of rows, number of columns)
tensor_2D.numel() # how many values in the FLATTENED array
tensor_2D[row_index, column_index] == tensor_2D[row_index][column_index] # equivalent
tensor_2D[3:5 rows, 1:3 columns]
linear_combination = 2*tensor_2D - 3.14*tensor_2D
elementwise_multiplication = tensor_2D * tensor_2D
matrix_multiplication = rows_of_A * columns_of_B = torch.mm(tensor_with_X_cols, tensor_with_X_rows)

numpy_tensor = pytorch_tensor.numpy(); pytorch_tensor = torch.from_numpy(numpy_tensor)
pandas_tensor = pandas.DataFrame({'a':[11,21,31],'b':[12,22,312]}); pytorch_tensor = torch.from_numpy(pandas_tensor.values)


#
# Derivatives
#
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
function = x * y + x ** 2
result_of_operations = function
function.backward()
partial_derivative_wrt_x = x.grad
partial_derivative_wrt_v = y.grad
print('data:',x.data)
print('grad_fn:',x.grad_fn)
print('grad:',x.grad)
print("is_leaf:",x.is_leaf)
print("requires_grad:",x.requires_grad)
print('data:',function.data)
print('grad_fn:',function.grad_fn)
print('grad:',function.grad)
print("is_leaf:",function.is_leaf)
print("requires_grad:",function.requires_grad)

class Custom_function_gradient(torch.autograd.Function):
# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input**3 + 2*input # what is the result of the operation = forward pass
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * ( 3*input**2 + 2 ) # what is the result of the derivative = backward pass
x = torch.tensor(3.0, requires_grad=True)
operation = Custom_function_gradient.apply
function = operation(x)
function.backward()
operations_result = function
derivative_wrt_x = x.grad
last_derivative = function.grad_fn

# Calculate the derivative with respect to a function with multiple values as follows.
# You use the sum trick to produce a scalar valued function and then take the gradient: 
# Calculate the derivative with multiple values
x = torch.linspace(-10, 10, 20, requires_grad=True)
Y = torch.relu(x) ** 2 # Take the derivative of Relu Squared with respect to multiple values.
# We need to explicitly pass a "gradient" argument in Q.backward() because it is a vector.
# "gradient" is a tensor of the same shape as Q, and it represents
# the gradient of Q w.r.t. itself, i.e. dQ/dQ = 1
# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
external_grad = torch.tensor([1., 1.]); Q.backward(gradient=external_grad)
# Equivalently, we can also aggregate Q into a scalar and call backward implicitly, like Q.sum().backward().
y = torch.sum(Y) # y = Y.sum()
y.backward()
x_values = x.detach().numpy()
y_values = Y.detach().numpy()
gradient_values = x.grad.detach().numpy()
plt.plot(x_values, y_values) # function
plt.plot(x_values, gradient_values) # derivative