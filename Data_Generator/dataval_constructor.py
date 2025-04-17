from . import diag_aligned_ele
from . import linear_aligned_ele
import numpy as np
import random

def dataval_constructor(number_of_value, pixel_diff = 0.5, greator_val = None, data_percent_split=0.5, shuffle = False):
    
    if greator_val == 'diagonal':
        diag_n =  int(number_of_value * data_percent_split)
        lin_n = number_of_value - diag_n
    elif greator_val == 'linear':
        lin_n = int(number_of_value * data_percent_split)
        diag_n = number_of_value - lin_n
    else:
        lin_n = int(number_of_value * data_percent_split)
        diag_n = number_of_value - lin_n    
        
 
    Lin_Values = linear_aligned_ele.linear_aligned_ele(pixel_diff, lin_n)
    Diag_Values = diag_aligned_ele.diag_aligned_ele(pixel_diff, diag_n)
    
    Full_List = Lin_Values + Diag_Values
    if shuffle == True:
        random.shuffle(Full_List)  

    X,y = list(zip(*Full_List))
     
    return np.array(X), np.array(y)

