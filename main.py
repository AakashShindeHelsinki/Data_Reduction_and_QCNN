import argparse
import Benchmarking


def main():
    #Could be hidden in local runs for multi runs / for slurm runs
    parser = argparse.ArgumentParser(description="QNN Training")
    parser.add_argument('--U',type=str,default='U_SU4',help='Unitary for QNN')
    parser.add_argument('--p',type=int,default=15, help='Parameters for the Unitrary')
    parser.add_argument('--rM',type=str, default='autoencode', help='Data Reduction Methods')
    parser.add_argument('--q',type=int, default=8, help='Number of Qubits')
    parser.add_argument('--em', type=str, default='Angle_X', help='Embedding Method')
    parser.add_argument('--data',type=str,default='sklearn_make_class', help='Dataset')
    
    args = parser.parse_args()
    print(f"Arguments Received : {args}")
    Unitaries = [args.U]
    U_num_params = [args.p]
    Embeddings = [[args.rM, args.q, args.em]]                     
    cost_fn = 'cross_entropy'
    data_gen = args.data
    
    """
    Unitaries = ['U_TTN','U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D','U_9_1D']
    U_num_params = [2,10,10,2,6,6,4,6,15,15,15,2]
    Embeddings = [['no_redu',16,'Angle_X'],['autoencode',16,'Angle_X'],['autoencode',14,'Angle_X'],['autoencode',12,'Angle_X'],['autoencode',10,'Angle_X'],['autoencode', 8, 'Angle_X']]#  
                                         #['no_redu',16,'Amplitude'],['pca',16,'Amplitude'],['pca',14,'Amplitude'],['pca',12,'Amplitude'],['pca',10,'Amplitude'],['pca',8,'Amplitude'],
                                         # ['no_redu',16,'Amplitude'],['autoencode',16,'Amplitude'],['autoencode',14,'Amplitude'],['autoencode',12,'Amplitude'],['autoencode',10,'Amplitude'],
    cost_fn = 'cross_entropy'
    data_gen = '4x4_img_data'
    """

    Benchmarking.Benchmarking(Unitaries, U_num_params, data_gen,Embeddings,cost_fn) 
    
if __name__ == "__main__":
    main()