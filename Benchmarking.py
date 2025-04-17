import data
import Training
import QCNN_circuit
import numpy as np
import csv
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score



#with scikitlearn
def evaluation(predictions, labels):
    accuracy = accuracy_score(predictions, labels)
    precision = precision_score(predictions, labels)
    f1 = f1_score(predictions, labels)
    recall = recall_score(predictions, labels)

    results = [accuracy,precision,f1,recall]
    return results

def Benchmarking(Unitaries, U_num_params, data_gen, Embedding,cost_fn):
    I = len(Unitaries)

    for i in range(I):
        for Embed in Embedding:
            U = Unitaries[i]
            U_params = U_num_params[i]

            X_train, Y_train, X_test, Y_test, DataID = data.data_load_and_process(q_num = Embed[1],data_gen = data_gen, data_redu=Embed[0])
            

            loss_history, trained_params, itter = Training.circuit_training(X_train, Y_train, U, U_params, q_num=Embed[1], embedding_type=Embed[2], cost_fn = cost_fn)


            predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params,Embed[1],Embed[2], cost_fn) for x in X_test]

            if cost_fn == 'cross_entropy':
                pred_array = []
                for pred in predictions:
                    if pred[0] > pred[1]:
                        pred_array.append(0)
                    else:
                        pred_array.append(1)
            else:
                for pred_a in range(0,len(predictions)):  
                    if predictions[pred_a] < 0:
                        predictions[pred_a] = 0
                    else:
                        predictions[pred_a] = 1
           


            evaluation_results = evaluation( pred_array, Y_test)
            #Print RESULTS HERE
            print(Embed)
            eva_res = ['accuracy','precision','f1','recall']
            for res,eva in zip(eva_res,evaluation_results):
                print(res+" : "+str(eva))


            print("\n Loss History :")
            print(loss_history)

            print("Updating CSV .... ")
            field_names = ['DataID','Model','DataPreProcessing','QbitNo','Encoding','Training_Itter','OptimisationAlgo','Lossfn','TrainParams','LossHist','Accuracy','Precision','F1','Recall']

            data_file = [{'DataID':DataID,'Model':U,'DataPreProcessing':Embed[0],'QbitNo':Embed[1],'Encoding':Embed[2],'Training_Itter':itter,
                          'OptimisationAlgo':'Nestrov Momentum','Lossfn':cost_fn,'TrainParams':trained_params,'LossHist':loss_history,
                          'Accuracy':evaluation_results[0],'Precision':evaluation_results[1],'F1':evaluation_results[2],'Recall':evaluation_results[3]}]

            with open('Results/TESTFILE_QNN_Binary_Synthetic.csv', 'a') as csvfile: 
                writer = csv.DictWriter(csvfile, fieldnames = field_names) 
                writer.writeheader() 
                writer.writerows(data_file)


