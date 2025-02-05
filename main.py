import Benchmarking

Unitaries = ['U_SU4_no_pooling']#, 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D','U_9_1D']
U_num_params = [15]#,10,10,2,6,6,4,6,15,15,15,2]
Embeddings = [['no_redu',16,'Amplitude'],['pca',16,'Amplitude'],['pca',14,'Amplitude'],['pca',12,'Amplitude'],['pca',10,'Amplitude'],['pca',8,'Amplitude']]
cost_fn = 'cross_entropy'
data_gen = 'capital1_synthetic_data'

Benchmarking.Benchmarking(Unitaries, U_num_params, data_gen,Embeddings,cost_fn)