import data
import Training
import QCNN_circuit

def dataset_creation_check():
    Embeddings = [['no_redu',16]]#,['pca',16],['pca',14],['pca',12],['pca',10],['pca',8]]
    for Embed in Embeddings:
        X_train, Y_train, X_test, Y_test, DataID = data.data_load_and_process(q_num = Embed[1], data_gen = '4x4_img_data' ,data_redu = Embed[0])
        print(f"Length of X_train value 0 : {len(X_train[0])}")
 
        print("X_train:")
        print(X_train)
        print("Y_train:")
        print(Y_train)
        print("X_test:")
        print(X_test)
        print("Y_test:")
        print(Y_test)
        print("DATA_ID:")
        print(DataID)
        print("NEXT Dataset Processing\n")

def Training_check():
    Embed = ['pca',8,'Amplitude']
    X_train, Y_train, X_test, Y_test, DataID = data.data_load_and_process(q_num = Embed[1], data_gen = 'sklearn_make_class' ,data_redu = Embed[0])
    loss_history, trained_params = Training.circuit_training(X_train, Y_train, U='U_9', U_params=2, q_num=Embed[1], 
                                                             embedding_type=Embed[2], cost_fn='cross_entropy')
    print("Training Check")
    print("Trained_params and Toss History")
    print(trained_params)
    print(loss_history)    

def Training_and_pred_check():
    Embed = ['pca',8,'Amplitude']
    X_train, Y_train, X_test, Y_test, DataID = data.data_load_and_process(q_num = Embed[1], data_gen = 'sklearn_make_class' , data_redu = Embed[0])
    loss_history, trained_params = Training.circuit_training(X_train, Y_train, U='U_SU4_no_pooling', U_params=15, q_num=Embed[1], 
                                                             embedding_type=Embed[2], cost_fn='cross_entropy')
    
    predictions = [QCNN_circuit.QCNN(x, trained_params, U='U_SU4_no_pooling', U_params=15,q_num=Embed[1],embedding_type=Embed[2], cost_fn='cross_entropy') for x in X_test]

    for pred in predictions:
        if pred[0] > pred[1]:
            pred = 0
        else:
            pred = 1
        print(pred)

    print("###################################")
    print(Y_test)



dataset_creation_check()
#Training_check()
#Training_and_pred_check()


