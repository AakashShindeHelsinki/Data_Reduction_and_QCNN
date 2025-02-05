from pennylane import AngleEmbedding, AmplitudeEmbedding

def data_embedding(X, q_num, type = 'Amplitude'):
    if type == 'Angle_X':
        AngleEmbedding(X, wires=range(q_num), rotation='X')
    elif type == 'Angle_Y':
        AngleEmbedding(X, wires=range(q_num), rotation='Y')
    elif type == 'Amplitude':
        AmplitudeEmbedding(X, wires=range(q_num), normalize=True, pad_with=1)