import random
import numpy as np

def check_val_list(num):
    five_list = [0,2,8,10]
    six_list = [1,3,9,11]
    nine_list = [4,6,12,14]
    ten_list = [5,7,13,15]
    
    if num == 5:
        num_two = random.choices(five_list)[0]
        num_three = int(random.choice([a for a in five_list if np.abs(a-num_two) == 2 or np.abs(a-num_two) == 8]))
        return  num_two, num_three
    elif num == 6:
        num_two = random.choices(six_list)[0]
        num_three = int(random.choice([a for a in six_list if np.abs(a-num_two)== 2 or np.abs(a-num_two)== 8]))
        return  num_two, num_three
    elif num == 9:
        num_two = random.choices(nine_list)[0]
        num_three = int(random.choice([a for a in nine_list if np.abs(a-num_two) == 2 or np.abs(a-num_two) == 8]))
        return  num_two, num_three
    elif num == 10:
        num_two = random.choices(ten_list)[0]
        num_three = int(random.choice([a for a in ten_list if np.abs(a-num_two)== 2 or np.abs(a-num_two) == 8]))
        return  num_two, num_three
    elif num in five_list:
        return 5, int(random.choice([a for a in five_list if np.abs(a-num) == 2 or np.abs(a-num) == 8]))
    elif num in six_list:
        six_list.remove(num)
        return 6,  int(random.choice([a for a in six_list if np.abs(a-num) == 2 or np.abs(a-num) == 8]))
    elif num in nine_list:
        nine_list.remove(num)
        return 9,  int(random.choice([a for a in nine_list if np.abs(a-num)== 2 or np.abs(a-num)== 8]))
    elif num in ten_list:
        ten_list.remove(num)
        return 10, int(random.choice([a for a in ten_list if np.abs(a-num) == 2 or np.abs(a-num) == 8]))
    
def diag_aligned_ele(diff, n):
    Diag_List = []
    for i in range(n):
        Diag_List_ele = []
        for j in range(16):
            random_num_w = np.round(random.uniform((1+diff)/2,1),4)
            Diag_List_ele.append(random_num_w)
            
        fir_val = random.randint(0,15)
        sec_val, third_val = check_val_list(fir_val)
        
        for i in [fir_val,sec_val,third_val]:
            Diag_List_ele[i] = np.round(random.uniform(0,(1-diff)/2),4)
            
        Diag_List.append([Diag_List_ele,0])    
    return Diag_List


    