import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(x, y, weights):
    x1, x2 = x[0], x[1]
    
    z1 = weights['w1'] * x1 + weights['w2'] * x2 + weights['b1']
    h1 = sigmoid(z1)
    z2 = weights['w3'] * x1 + weights['w4'] * x2 + weights['b1']
    h2 = sigmoid(z2)
    
    z3 = weights['w5'] * h1 + weights['w6'] * h2 + weights['b2']
    o1 = sigmoid(z3)
    z4 = weights['w7'] * h1 + weights['w8'] * h2 + weights['b2']
    o2 = sigmoid(z4)
    
    output = np.array([o1, o2])
    loss = np.mean((output - y) ** 2)
    
    forward_info = {
        'x1': x1, 'x2': x2,
        'z1': z1, 'h1': h1,
        'z2': z2, 'h2': h2,
        'z3': z3, 'o1': o1,
        'z4': z4, 'o2': o2
    }
    return forward_info, loss

def backward(forward_info, y, weights, learning_rate=0.01):

    x1 = forward_info['x1']
    x2 = forward_info['x2']
    h1 = forward_info['h1']
    h2 = forward_info['h2']
    o1 = forward_info['o1']
    o2 = forward_info['o2']

    dL_do1 = o1 - y[0]
    d_o1_dz3 = o1 * (1 - o1)
    dL_dz3 = dL_do1 * d_o1_dz3
    
    dL_do2 = o2 - y[1]
    d_o2_dz4 = o2 * (1 - o2)
    dL_dz4 = dL_do2 * d_o2_dz4
    
    grad_w5 = dL_dz3 * h1
    grad_w6 = dL_dz3 * h2
    grad_w7 = dL_dz4 * h1
    grad_w8 = dL_dz4 * h2

    grad_b2 = dL_dz3 + dL_dz4
    
   
    dL_dh1 = dL_dz3 * weights['w5'] + dL_dz4 * weights['w7']
    dL_dh2 = dL_dz3 * weights['w6'] + dL_dz4 * weights['w8']
    
    d_h1_dz1 = h1 * (1 - h1)
    dL_dz1 = dL_dh1 * d_h1_dz1
    
    d_h2_dz2 = h2 * (1 - h2)
    dL_dz2 = dL_dh2 * d_h2_dz2
    
    grad_w1 = dL_dz1 * x1
    grad_w2 = dL_dz1 * x2
    grad_w3 = dL_dz2 * x1
    grad_w4 = dL_dz2 * x2

    grad_b1 = dL_dz1 + dL_dz2
    
    weights['w1'] -= learning_rate * grad_w1
    weights['w2'] -= learning_rate * grad_w2
    weights['w3'] -= learning_rate * grad_w3
    weights['w4'] -= learning_rate * grad_w4
    weights['b1'] -= learning_rate * grad_b1
    weights['w5'] -= learning_rate * grad_w5
    weights['w6'] -= learning_rate * grad_w6
    weights['w7'] -= learning_rate * grad_w7
    weights['w8'] -= learning_rate * grad_w8
    weights['b2'] -= learning_rate * grad_b2

    gradients = {
        'w1': grad_w1, 'w2': grad_w2, 'w3': grad_w3, 'w4': grad_w4,
        'b1': grad_b1, 'w5': grad_w5, 'w6': grad_w6,
        'w7': grad_w7, 'w8': grad_w8, 'b2': grad_b2
    }
    return gradients


weights = {
    'w1': 0.15, 'w2': 0.20,
    'w3': 0.25, 'w4': 0.30,
    'b1': 0.35,
    'w5': 0.40, 'w6': 0.45,
    'w7': 0.50, 'w8': 0.55,
    'b2': 0.60
}


x = np.array([0.05, 0.10])
y = np.array([0.01, 0.99])


forward_info, loss = forward(x, y, weights)
print("Loss:", loss)

gradients = backward(forward_info, y, weights)
print("Updated weights:", weights)
