# Implementation of Quantum circuit training procedure
import QCNN_circuit
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss

def cost(params, X, Y, U, U_params, q_num,embedding_type, cost_fn='mse'):
        
    predictions = [QCNN_circuit.QCNN(x, params, U, U_params, q_num, embedding_type, cost_fn=cost_fn) for x in X]
   
    if cost_fn == 'mse':
        loss = square_loss(Y, predictions)
    elif cost_fn == 'cross_entropy':
        loss = cross_entropy(Y, predictions)
    return loss

# Circuit training parameters
steps = 300
learning_rate = 0.01
batch_size = 15
def circuit_training(X_train, Y_train, U, U_params, q_num, embedding_type, cost_fn):
    if q_num >=10 :
        if U == 'U_SU4_no_pooling' or U == 'U_SU4_1D' or U == 'U_9_1D':
            total_params = U_params * 4
        else:
            total_params = U_params * 4 + 2 * 4
    else:
        if U == 'U_SU4_no_pooling' or U == 'U_SU4_1D' or U == 'U_9_1D':
            total_params = U_params * 3
        else:
            total_params = U_params * 3 + 2 * 3
  

    params = np.random.randn(total_params, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    loss_history = []

    for it in range(steps):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, q_num, embedding_type,cost_fn), params)
        loss_history.append(cost_new)
        if (it+1) % 10 == 0:
            print("iteration: ", it+1, " cost: ", cost_new)
    return loss_history, params, steps

