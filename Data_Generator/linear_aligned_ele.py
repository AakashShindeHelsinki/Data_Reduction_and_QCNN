import random

def two_list_Val(num):
    if num == 0:
        return 1,4
    elif num == 3:
        return 2,7
    elif num == 12:
        return 8,13
    else:
        return 11,14
    
def three_num_list_ver_Val(num):
    if num == 1:
        return 5, random.choice([0,2])
    elif num == 2:
        return 6, random.choice([1,3])
    elif num == 13:
        return 9, random.choice([12,14])
    elif num == 14:
        return 10, random.choice([13,15])
    

def three_num_list_hor_Val(num):
    if num == 4:
        return random.choice([0,8]), 5 
    elif num == 7:
        return random.choice([3,4]), 6
    elif num == 8:
        return random.choice([4,12]), 9
    elif num == 11:
        return random.choice([7,15]), 10

def four_num_list_Val(num):
    if num == 5:
        return random.choice([4,6]), random.choice([1,9]) 
    elif num == 6:
        return random.choice([5,7]), random.choice([2,10])
    elif num == 9:
        return random.choice([8,10]), random.choice([5,13])
    elif num == 10:
        return random.choice([9,11]), random.choice([6,14])
    
    
def check_val_list(num):
    two_num_list = [0,3,12,15]
    three_num_list_ver = [1,2,13,14]
    three_num_list_hor = [4,7,8,11]
    four_num_list = [5,6,9,10]
    
    if num in two_num_list:
        return two_list_Val(num)
    elif num in three_num_list_ver:
        return three_num_list_ver_Val(num)
    elif num in three_num_list_hor:
        return three_num_list_hor_Val(num)
    elif num in four_num_list:
        return four_num_list_Val(num)
    
        

def linear_aligned_ele(diff, n):
    Lin_List = []
    for i in range(n):
        Lin_List_ele = []
        for j in range(16):
            random_num_w = round(random.uniform((1+diff)/2,1),4)
            Lin_List_ele.append(random_num_w)
            
        fir_val = random.randint(0,15)
        sec_val, third_val = check_val_list(fir_val)
        for i in [fir_val,sec_val,third_val]:
            Lin_List_ele[i] = round(random.uniform(0,(1-diff)/2),4)
             
        Lin_List.append([Lin_List_ele,1])
        
    return Lin_List

