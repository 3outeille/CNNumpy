from src.fast.layers import Conv, AvgPool
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(42)
torch.set_printoptions(precision=8)

test_registry = {}
def register_test(func):
    test_registry[func.__name__] = func
    return func

@register_test
def test_conv():
    n, c, h, w = 20, 10, 5, 5 # Image.
    nf, cf, hf, wf = 2, 10, 3, 3 # Filters.
    x = np.random.randn(n, c, h, w)
    x = x.astype('float64')
    W = np.random.randn(nf, cf, hf, wf)
    W = W.astype('float64')
    b = np.random.randn(nf)
    b = b.astype('float64')
    deltaL = np.random.rand(n, nf, h-hf+1, h-hf+1) # Assuming stride=1/padding=0.

    # CNNumpy.
    inputs_cnn = x
    weights_cnn, biases_cnn = W, b
    conv_cnn = Conv(nb_filters=nf, filter_size=hf, nb_channels=cf, stride=1, padding=0)
    conv_cnn.W['val'], conv_cnn.W['grad'] = weights_cnn, np.zeros_like(weights_cnn)
    conv_cnn.b['val'], conv_cnn.b['grad'] = biases_cnn, np.zeros_like(biases_cnn)

    out_cnn = conv_cnn.forward(inputs_cnn) # Forward.
    _, conv_cnn.W['grad'], conv_cnn.b['grad'] = conv_cnn.backward(deltaL) # Backward.

    # Pytorch
    inputs_pt = torch.Tensor(x).double()
    conv_pt = nn.Conv2d(c, nf, hf, stride=1, padding=0, bias=True)
    conv_pt.weight = nn.Parameter(torch.DoubleTensor(W))
    conv_pt.bias = nn.Parameter(torch.DoubleTensor(b))
    out_pt = conv_pt(inputs_pt) # Forward.
    out_pt.backward(torch.Tensor(deltaL)) # Backward.

    # Check if inputs are equals.
    assert np.allclose(inputs_cnn, inputs_pt.numpy(), atol=1e-8) # 1e-8 tolerance.
    # Check if weights are equals.
    assert np.allclose(weights_cnn, conv_pt.weight.data.numpy(), atol=1e-8) # 1e-8 tolerance.
    # Check if biases are equals.
    assert np.allclose(biases_cnn, conv_pt.bias.data.numpy(), atol=1e-8) # 1e-8 tolerance.
    # Check if conv forward outputs are equals. 
    assert np.allclose(out_cnn, out_pt.data.numpy(), atol=1e-5) # 1e-5 tolerance.
    # Check if conv backward outputs are equals. 
    assert np.allclose(conv_cnn.W['grad'], conv_pt.weight.grad.numpy(), atol=1e-5) # 1e-5 tolerance.
    assert np.allclose(conv_cnn.b['grad'], conv_pt.bias.grad.numpy(), atol=1e-5) # 1e-5 tolerance.

@register_test
def test_avgpool():
    n, c, h, w = 10, 4, 5, 5 # Image.
    kernel_size, stride = 2, 1
    x = np.random.randn(n, c, h, w)
    # x = np.arange(n*c*h*w).reshape(n,c,h,w)
    # x += 1
    x = x.astype('float64')
    H = int((h - kernel_size)/ stride) + 1
    W = int((w - kernel_size)/ stride) + 1
    deltaL = np.random.rand(n, c, H, W) 
    # deltaL = np.arange(n*c*H*W).reshape(n,c,H,W)
    # deltaL += 1

    # CNNumpy.
    inputs_cnn = x
    avg_cnn = AvgPool(kernel_size, stride=stride)
    out_cnn = avg_cnn.forward(inputs_cnn) # Forward.
    inputs_cnn_grad = avg_cnn.backward(deltaL) # Backward.

    # Pytorch.
    inputs_pt = torch.Tensor(x).double()
    inputs_pt.requires_grad = True
    avg_pt = nn.AvgPool2d(kernel_size, stride = stride)
    out_pt = avg_pt(inputs_pt) # Forward.
    out_pt.backward(torch.Tensor(deltaL)) # Backward.

    # Check if inputs are equals.
    assert np.allclose(inputs_cnn, inputs_pt.data.numpy(), atol=1e-8) # 1e-8 tolerance.
    # Check if conv forward outputs are equals. 
    assert np.allclose(out_cnn, out_pt.data.numpy(), atol=1e-8) # 1e-8 tolerance.
    # Check if conv backward outputs are equals. 
    assert np.allclose(inputs_cnn_grad, inputs_pt.grad.numpy(), atol=1e-7) # 1e-7 tolerance.

for name, test in test_registry.items():
    print(f'Running {name}:', end=" ")
    test()
    print("OK")