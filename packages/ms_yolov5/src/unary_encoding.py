"""Implements unary encoding, used by Google's RAPPOR, as the local differential privacy mechanism.
References:
Wang, et al. "Optimizing Locally Differentially Private Protocols," ATC USENIX 2017.
Erlingsson, et al. "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response,"
ACM CCS 2014."""

import math
import numpy as np
import mindspore 
from mindspore import Tensor
import mindspore.numpy as mnp
import mindspore as ms

def encode_1b(x):
    #print("1 bit encoded")
    x[(x <= 0)] = 0
    x[(x > 0)] = 1
    return x
def randomize_1b(bit_tensor, epsilon):
    """
    The default unary encoding method is symmetric.
    """
    #assert isinstance(bit_tensor, tensor), 'the type of input data is not matched with the expected type(tensor)'
    return symmetric_tensor_encoding_1b(bit_tensor, epsilon)
    
def symmetric_tensor_encoding_1b(bit_tensor, epsilon):
    #epsilon = 100
    p = mnp.exp(epsilon / 2) / (mnp.exp(epsilon / 2) + 1)
    q = 1 / (mnp.exp(epsilon / 2) + 1)
    #print("1bit randomized")
    return produce_random_response_1b(bit_tensor, p, q)
    
def produce_random_response_1b(bit_tensor, p, q=None):
    """
    Implements random response as the perturbation method.
    when using torch tensor, we use Uniform Distribution to create Binomial Distribution
    because torch have not binomial function
    """
    q = 1 - p if q is None else q
    uniformreal = mindspore.ops.UniformReal(seed=2)
    binomial = uniformreal(bit_tensor.shape)
    zeroslike = mindspore.ops.ZerosLike()
    oneslike = mindspore.ops.OnesLike()
    p_binomial = mnp.where(binomial > q, oneslike(bit_tensor), zeroslike(bit_tensor))
    q_binomial = mnp.where(binomial <= q, oneslike(bit_tensor), zeroslike(bit_tensor))
    return mnp.where(bit_tensor == 1, p_binomial, q_binomial)   
    
    
    
        
def encode_2b(x):
    #print("2 bit encoded")
    #x = x.asnumpy()
    x[(x > 0.675)] = 1.15 
    #x[(x > 0) & (x <= 0.675)] = 0.315
    #x[(x > -0.675) & (x <= 0)] = -0.315
    x[(x <= -0.675)] = -1.15
    #x = Tensor(x)
    return x
                    
    

def randomize_2b(bit_tensor, epsilon):
    """
    The default unary encoding method is symmetric.
    """
    #assert isinstance(bit_tensor, tensor), 'the type of input data is not matched with the expected type(tensor)'
    return symmetric_tensor_encoding_2b(bit_tensor, epsilon)

def symmetric_tensor_encoding_2b(bit_tensor, epsilon):
    p = mnp.exp(epsilon / 2) / (mnp.exp(epsilon / 2) + 3)
    q = (1-p)/3 #round((1-p)/3, 4)
    p = 1 - q * 3
    #print("2bit randomized")
    L = [-1.15, -0.315, 0.315, 1.15]
    return k_random_response_2b(bit_tensor, L, p, q)
   
def k_random_response_2b(value, values, p, q):

    #if not isinstance(values, list):
    #    raise Exception("The values should be list")

    p = (p-0.25)*4/3

    change = np.random.choice(values, size=value.shape, replace=True, p=[0.25,0.25,0.25,0.25])
    uniformreal = mindspore.ops.UniformReal(seed=2)
    binomial = uniformreal(value.shape)
    out_tensor = mnp.where(binomial < p, value, Tensor(change).astype(mnp.float32))
    return out_tensor
    
##########################################################    
def optimized_mindspore_encoding(bit_tensor, epsilon):
    p = 1 / 2
    q = 1 / (math.exp(epsilon) + 1)
    return produce_random_response(bit_tensor, p, q)
def randomize_box(bit_array, targets, epsilon):
    """
    The object detection unary encoding method.
    """
    assert isinstance(bit_array, tensor)
    img = symmetric_tensor_encoding(bit_array, 1)
    label = symmetric_tensor_encoding(bit_array, epsilon)
    targets_new = targets.clone()
    targets_new = targets_new.detach().numpy()
    for i in range(targets_new.shape[1]):
        box = convert(bit_array.shape[2:], targets_new[i][2:])
        img[:, :, box[0]:box[2],
            box[1]:box[3]] = label[:, :, box[0]:box[2], box[1]:box[3]]
    return img

def convert(size, box):
    """The convert for YOLOv5.
            Arguments:
                size: Input feature size(w,h)
                box:(xmin,xmax,ymin,ymax).
            """
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    x1 = max(x - 0.5 * w - 3, 0)
    x2 = min(x + 0.5 * w + 3, size[0])
    y1 = max(y - 0.5 * h - 3, 0)
    y2 = min(y + 0.5 * h + 3, size[1])

    x1 = round(x1 * size[0])
    x2 = round(x2 * size[0])
    y1 = round(y1 * size[1])
    y2 = round(y2 * size[1])

    return (int(x1), int(y1), int(x2), int(y2))


if __name__ in '__main__':
    stdnormal = mindspore.ops.StandardNormal(seed=2)
    shape = (1,2,8, 8)
    logits = Tensor(stdnormal(shape).astype(mnp.float32))
    #logits=Tensor(mnp.arange(8 * 8).reshape((8, 8)).astype(mnp.float32))
    print(logits)
    logits = encode_2b(logits)
    print('dtype:',logits.dtype)
    logitss = randomize_2b(logits, 100)
    print(logitss)
    print('dtype:',logitss.dtype)

