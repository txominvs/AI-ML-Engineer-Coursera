import torch

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