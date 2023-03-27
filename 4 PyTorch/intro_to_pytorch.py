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
print('data:',x.data) # where to evaluate
print('grad_fn:',x.grad_fn) # NONE
print('grad:',x.grad) # partial derivative wrt X
print("is_leaf:",x.is_leaf) # yes, because X is a the top of the backwards propagation
print("requires_grad:",x.requires_grad) # yes
print('data:',function.data) # result of math operations
print('grad_fn:',function.grad_fn) # gradient of the last executed operation, first in the backwards propagation chain
print('grad:',function.grad) # NONE
print("is_leaf:",function.is_leaf) # nope, actually this is the base trunk of the backwards propagation
print("requires_grad:",function.requires_grad) # yes, inhertied

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


#
# Dataset
#
from torch.utils.data import Dataset

class Custom_dataset(Dataset):
    def __init__(self, length=100, transform=None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform
    def __getitem__(self, index): # overrides Custom_dataset()[index]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self): # overrides len( Custom_dataset() )
        return self.len

class Custom_transformation():
    def __init__(self, add_smth=0): # setup
        self.add_smth = add_smth
        pass
    def __call__(self, sample): # perform transformation
        x, y = sample
        x = x + self.add_smth
        return x, y

dataset = Custom_dataset(
    length=10,
    transform=Custom_transformation(add_smth=123),
)

from torchvision import transforms
composed_transformation = transforms.Compose([
    Custom_transformation(add_smth=123),
    Custom_transformation(add_smth=-1),
])

from torch.utils.data import Dataloader
train_loader = Dataloader(dataset=dataset, batch_size=5)
for epoch in range(100):
    for batch_x, batch_y in train_loader:
        ...

#
# On-demand datasets
#
class Dataset(Dataset):
    def __init__(self, database_location):
        self.data_base = pandas.read_csv(database_location)
        self.len = self.data_base.shape[0] 
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        img_location = os.path.join(base_url, self.data_base.iloc[index, 1])
        image = Image.open(img_location)
        image = self.transform(image)
        label = self.data_base.iloc[idx, 0]
        return image, label
import torchvision.transforms as transforms
transformations = transforms.Compose([
    transforms.CenterCrop(20), # crop a 20x20 square in the middle
    transforms.RandomVerticalFlip(p=1),
    transforms.RandomHorizontalFlip(p=1)
    transforms.ToTensor()
])
dataset = Dataset(
    database_location="something.csv",
    transform = transformations,
)

import torchvision.datasets as dsets
dataset = dsets.MNIST(
    root='./data',
    train = False, # download training set or test set?
    download = True,
    transform = transformations,
) # = images shape 1x28x28 , labels

