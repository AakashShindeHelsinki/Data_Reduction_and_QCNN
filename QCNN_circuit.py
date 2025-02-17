import pennylane as qml
import Ansatz
import Embedding 

#change if necessary
#Convolutional
def Conv16q(U, params):
    for i in range(0,8):
        U(params,wires=[i,15-i])
    for i in range(0,7):
        U(params,wires=[i,14-i])
    U(params,wires=[7,15])

def Conv8q(U, params):
    for i in range(0,4):
        U(params, wires=[i,7-i])
    for i in range(0,3):
        U(params, wires=[i,6-i])
    U(params, wires=[3,4])

def Conv4q(U, params):
    U(params, wires=[0,3])
    U(params, wires=[1,2])
    U(params, wires=[0,2])
    U(params, wires=[1,3])

def Conv2q(U, params):
    U(params, wires=[0,1])    


def Conv14q(U, params):
    for i in range(0,7):
        U(params,wires=[i,13-i])
    for i in range(0,6):
        U(params,wires=[i,12-i])
    U(params,wires=[6,13])

def Conv12q(U, params):
    for i in range(0,6):
        U(params,wires=[i,11-i])
    for i in range(0,5):
        U(params,wires=[i,10-i])
    U(params,wires=[5,11])

def Conv6q(U, params):
    for i in range(0,3):
        U(params,wires=[i,5-i])
    for i in range(0,2):
        U(params,wires=[i,4-i])
    U(params,wires=[2,5])

def Conv10q(U, params):
    for i in range(0,5):
        U(params,wires=[i,9-i])
    for i in range(0,4):
        U(params,wires=[i,8-i])
    U(params,wires=[4,9])



#Pooling Structures
def pooling16q(V, params):
    for i in range(0,8):
        V(params,wires=[15-i,i])

def pooling8q(V, params):
    for i in range(0,4):
        V(params,wires=[7-i,i])

def pooling4q(V, params):
    V(params,wires=[3,0])
    V(params,wires=[2,1])

def pooling2q(V, params):
    V(params,wires=[1,0])

def pooling14q(V, params):
    for i in range(0,7):
        V(params, wires=[13-i,i])

def pooling12q(V, params):
    for i in range(0,6):
        V(params, wires=[11-i,i])

def pooling10q(V, params):
    for i in range(0,5):
        V(params, wires=[9-i,i])

def pooling6q(V, params):
    V(params,wires=[5,0])
    V(params,wires=[4,1])
    V(params,wires=[3,2])
    
#########################################################################################################

#Full Structure of Quantum Circuit
def QCNN_struct_16q(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]
    param5 = params[4 * U_params: 4 * U_params + 2]
    param6 = params[4 * U_params + 2: 4 * U_params + 4]
    param7 = params[4 * U_params + 4: 4 * U_params + 6]
    param8 = params[4 * U_params + 6: 4 * U_params + 8]


    #Combining Ansatz
    Conv16q(U, param1)
    pooling16q(Ansatz.pooling_ansatz,param5)
    Conv8q(U, param2)
    pooling8q(Ansatz.pooling_ansatz,param6)
    Conv4q(U, param3)
    pooling4q(Ansatz.pooling_ansatz,param7)
    Conv2q(U, param4)
    pooling2q(Ansatz.pooling_ansatz,param8)


def QCNN_struct_14q(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]
    param5 = params[4 * U_params: 4 * U_params + 2]
    param6 = params[4 * U_params + 2: 4 * U_params + 4]
    param7 = params[4 * U_params + 4: 4 * U_params + 6]
    param8 = params[4 * U_params + 6: 4 * U_params + 8]
    #Combining Ansatz
    Conv14q(U, param1)
    pooling14q(Ansatz.pooling_ansatz,param5)
    Conv8q(U, param2)
    pooling8q(Ansatz.pooling_ansatz,param6)
    Conv4q(U, param3)
    pooling4q(Ansatz.pooling_ansatz,param7)
    Conv2q(U, param4)
    pooling2q(Ansatz.pooling_ansatz,param8)

def QCNN_struct_12q(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]
    param5 = params[4 * U_params: 4 * U_params + 2]
    param6 = params[4 * U_params + 2: 4 * U_params + 4]
    param7 = params[4 * U_params + 4: 4 * U_params + 6]
    param8 = params[4 * U_params + 6: 4 * U_params + 8]
    #Combining Ansatz
    Conv12q(U, param1)
    pooling12q(Ansatz.pooling_ansatz,param5)
    Conv6q(U, param2)
    pooling6q(Ansatz.pooling_ansatz,param6)
    Conv4q(U, param3)
    pooling4q(Ansatz.pooling_ansatz,param7)
    Conv2q(U, param4)
    pooling2q(Ansatz.pooling_ansatz,param8) 

def QCNN_struct_10q(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]
    param5 = params[4 * U_params: 4 * U_params + 2]
    param6 = params[4 * U_params + 2: 4 * U_params + 4]
    param7 = params[4 * U_params + 4: 4 * U_params + 6]
    param8 = params[4 * U_params + 6: 4 * U_params + 8]
    #Combining Ansatz
    Conv10q(U, param1)
    pooling10q(Ansatz.pooling_ansatz,param5)
    Conv6q(U, param2)
    pooling6q(Ansatz.pooling_ansatz,param6)
    Conv4q(U, param3)
    pooling4q(Ansatz.pooling_ansatz,param7)
    Conv2q(U, param4)
    pooling2q(Ansatz.pooling_ansatz,param8) 

def QCNN_struct_8q(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 6]

    #Combining Ansatz
    Conv8q(U, param1)
    pooling8q(Ansatz.pooling_ansatz,param4)
    Conv4q(U, param2)
    pooling4q(Ansatz.pooling_ansatz,param5)
    Conv2q(U, param3)
    pooling2q(Ansatz.pooling_ansatz,param6)

####################################################################################################

def QCNN_struct_16q_1D(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]

    #Combining Ansatz
    for i in range(0,8):
        U(param1,wires=[i,15-i])
    for i in range(0,7):
        U(param1,wires=[i,14-i])

    for i in range(0,4):
        U(param2, wires=[i,7-i])
    for i in range(0,3):
        U(param2, wires=[i,6-i])

    U(param3, wires=[0,3])
    U(param3, wires=[1,2])

    U(param4, wires=[1,0])  
    

def QCNN_struct_14q_1D(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]

    #Combining Ansatz
    for i in range(0,7):
        U(param1,wires=[i,13-i])
    for i in range(0,6):
        U(param1,wires=[i,12-i])

    
    for i in range(0,4):
        U(param2, wires=[i,7-i])
    for i in range(0,3):
        U(param2, wires=[i,6-i])

    U(param3, wires=[0,3])
    U(param3, wires=[1,2])

    U(param4, wires=[1,0])  


def QCNN_struct_12q_1D(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]

    #Combining Ansatz
    for i in range(0,6):
        U(param1,wires=[i,11-i])
    for i in range(0,5):
        U(param1,wires=[i,10-i])

    for i in range(0,3):
        U(param2,wires=[i,5-i])
    for i in range(0,2):
        U(param2,wires=[i,4-i])

    
    U(param3, wires=[0,3])
    U(param3, wires=[1,2])

    U(param4, wires=[1,0]) 

def QCNN_struct_10q_1D(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]

    #Combining Ansatz
    for i in range(0,5):
        U(param1,wires=[i,9-i])
    for i in range(0,4):
        U(param1,wires=[i,8-i])

    for i in range(0,3):
        U(param2,wires=[i,5-i])
    for i in range(0,2):
        U(param2,wires=[i,4-i])
    
    U(param3, wires=[0,3])
    U(param3, wires=[1,2])

    U(param4, wires=[1,0]) 

def QCNN_struct_8q_1D(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]


    #Combining Ansatz
    for i in range(0,4):
        U(param1, wires=[i,7-i])
    for i in range(0,3):
        U(param1, wires=[i,6-i])

    U(param2, wires=[0,3])
    U(param2, wires=[1,2])

    U(param3, wires=[1,0])
#########################################################################################################

def QCNN_struct_16q_no_pooling(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]

    #Combining Ansatz
    Conv16q(U, param1)
    Conv8q(U, param2)
    Conv4q(U, param3)
    Conv2q(U, param4)


def QCNN_struct_14q_no_pooling(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]

    #Combining Ansatz
    Conv14q(U, param1)
    Conv8q(U, param2)
    Conv4q(U, param3)
    Conv2q(U, param4)


def QCNN_struct_12q_no_pooling(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]

    #Combining Ansatz
    Conv12q(U, param1)
    Conv6q(U, param2)
    Conv4q(U, param3)
    Conv2q(U, param4)


def QCNN_struct_10q_no_pooling(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]

    #Combining Ansatz
    Conv10q(U, param1)
    Conv6q(U, param2)
    Conv4q(U, param3)
    Conv2q(U, param4)

def QCNN_struct_8q_no_pooling(U, params, U_params):
    #Params
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]

    #Combining Ansatz
    Conv8q(U, param1)
    Conv4q(U, param2)
    Conv2q(U, param3)


#########################################################################################################



#Check with the way of addressing this issue from bechmarking perspective

def QCNN(x, params, U, U_params,q_num = 16,embedding_type='Amplitude',cost_fn='cross_entropy'):
    #Choose Model here
    Ansatz_Dict = {'U_TTN': Ansatz.U_TTN, 'U_5': Ansatz.U_5, 'U_6':Ansatz.U_6, 'U_9':Ansatz.U_9, 'U_13':Ansatz.U_13,
                    'U_14':Ansatz.U_14, 'U_15':Ansatz.U_15, 'U_SO4': Ansatz.U_SO4, 'U_SU4':Ansatz.U_SU4,
                    'U_SU4_no_pooling':Ansatz.U_SU4, 'U_SU4_1D':Ansatz.U_SU4,'U_9_1D':Ansatz.U_9}

    Anz = Ansatz_Dict[U]


    if q_num == 16:
        if U == 'U_SU4_no_pooling':
            Q_model = 'Conv16_np'
        elif U == 'U_SU4_1D' or U == 'U_9_1D':
            Q_model = 'Conv16_1D'
        else:
            Q_model = 'Conv16'
    elif q_num == 14:
        if U == 'U_SU4_no_pooling':
            Q_model = 'Conv14_np'
        elif U == 'U_SU4_1D' or U =='U_9_1D':
            Q_model = 'Conv14_1D'
        else:
            Q_model = 'Conv14'
    elif q_num == 12:
        if U == 'U_SU4_no_pooling':
            Q_model = 'Conv12_np'
        elif U == 'U_SU4_1D' or U =='U_9_1D':
            Q_model = 'Conv12_1D'
        else:
            Q_model = 'Conv12'
    elif q_num == 10:
        if U == 'U_SU4_no_pooling':
            Q_model = 'Conv10_np'
        elif U == 'U_SU4_1D' or U =='U_9_1D':
            Q_model = 'Conv10_1D'
        else:
            Q_model = 'Conv10'
    elif q_num == 8:
        if U == 'U_SU4_no_pooling':
            Q_model = 'Conv8_np'
        elif U == 'U_SU4_1D' or U=='U_9_1D':
            Q_model = 'Conv8_1D'
        else:
            Q_model = 'Conv8'






    dev = qml.device('default.qubit', wires = q_num)
    @qml.qnode(dev)
    def Circuit_Run(x,params, Anz, Q_model,U_params,q_num,embedding_type,cost_fn):
        #DATA Embedding
        Embedding.data_embedding(x, q_num, type=embedding_type)
        if Q_model == 'Conv16_np':
            QCNN_struct_16q_no_pooling(Anz, params, U_params)
        elif Q_model == 'Conv16_1D':
             QCNN_struct_16q_1D(Anz, params, U_params)
        elif Q_model == 'Conv16':
            QCNN_struct_16q(Anz, params, U_params)
        elif Q_model == 'Conv14_np':
            QCNN_struct_14q_no_pooling(Anz, params, U_params)
        elif Q_model == 'Conv14_1D':
            QCNN_struct_14q_1D(Anz, params, U_params)
        elif Q_model == 'Conv14':
            QCNN_struct_14q(Anz, params, U_params)
        elif Q_model == 'Conv12_np':
            QCNN_struct_12q_no_pooling(Anz, params, U_params)
        elif Q_model == 'Conv12_1D':
            QCNN_struct_12q_1D(Anz, params, U_params)
        elif Q_model == 'Conv12':
            QCNN_struct_12q(Anz, params, U_params)
        elif Q_model == 'Conv10_np':
            QCNN_struct_10q_no_pooling(Anz, params, U_params)
        elif Q_model == 'Conv10_1D':
            QCNN_struct_10q_1D(Anz, params, U_params)
        elif Q_model == 'Conv10':
            QCNN_struct_10q(Anz, params, U_params)
        elif Q_model == 'Conv8_np':
            QCNN_struct_8q_no_pooling(Anz, params, U_params)
        elif Q_model == 'Conv8_1D':
            QCNN_struct_8q_1D(Anz, params, U_params)
        elif Q_model == 'Conv8':
            QCNN_struct_8q(Anz, params, U_params)
        else:
            print("Wrong Ansatz Request")
            return False

        if cost_fn == 'mse':
            result = qml.expval(qml.PauliZ((0)))
        elif cost_fn == 'cross_entropy':
            result = qml.probs(wires=0)
        return result
    
    return Circuit_Run(x, params, Anz, Q_model, U_params, q_num, embedding_type, cost_fn)
    
    


