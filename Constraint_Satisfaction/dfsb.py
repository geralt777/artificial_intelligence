import sys
import math
import time
import random
import heapq
from copy import deepcopy
import numpy as np
from collections import defaultdict
from collections import deque
import copy
from operator import itemgetter

n=0
m=0
k=0

#function to select an unassigned variable based on the MRV heuristic
#assignments is a list representing the state of the current assignment
#domain_vals is a list containing the remaining legal values of each variable at the time of the function call
#we return an unassigned variable with the least number of legal values
def select_unassigned_var(assignments, domain_vals):
    minval = 9999
    retval = 9999
    for i in range(0,n):
        val_comp = len(domain_vals[i])
        if((assignments[i]==-1) and (val_comp < minval)):
            minval = val_comp
            retval = i
    return retval
    #max_dom_val = len(domain_vals[0])
    
#function to order domain values
#constraints as given by the input
#variable is the variable for which we have to decide the best possible value based on current assignments
#return a custom sorted list of domain values for the given variable
def order_domain_vals(constraints, assignments, variable, domain_vals):
    #empty list of tuples to return
    new_list = []
    for val in domain_vals[variable]:
        #let the current assignment of the variable be val
        #we'll reduce the domains of all variables in the constraints list of the current variable
        min_val = 9999
        num_red = 0
        for j in constraints[variable]:
            #check for the variables that haven't been assigned yet
            #if(assignments[j]!=-1):
                #This value has already been assigned
                #continue
            #calculate the number of reductions to the domains of the remaining unassigned neighbors
            if(val in domain_vals[j]):
                num_red+=1
        pair = (val, num_red)
        new_list.append(pair)
    #sort list based on the minval for each tuple, min value first
    fin_list = sorted(new_list, key=itemgetter(1))
    return fin_list
 
#check and apply arc consistency 
def ac3_consistency(constraints, domain_values, tup_list_next):
    #create a new queue
    myq = deque([])
    #get all the arcs from the constraints
    for vals in tup_list_next:
        myq.append(vals)
        
    #print("1")
    #print(domain_values)
            
    while((len(myq))>0):
        cur_tup = myq.popleft()
        xi = cur_tup[0]
        xj = cur_tup[1]
        #revise (if required) the domain of the popped key
        revised = 0
        #temp_xi_list = domain_values[xi].copy()
        rem_arr = []
        for i in domain_values[xi]:
            #check for constraint satisfaction
            temp = 0
            for j in domain_values[xj]:
                if (j!=i):
                    temp = 1
                    break;
            #if no valid value was found...
            if(temp==0):
                revised = 1
                #temp_list[xi].remove(i)
                #mark for removal
                rem_arr.append(i)
        
        #print("LEN")
        #print(len(rem_arr))
        for i in rem_arr:
            domain_values[xi].remove(i)
        
        if(revised == 1):
            global total_prunes
            total_prunes+=1
            #print("Revised")
            #domain_values[xi] = temp_xi_list
            if((len(domain_values[xi]))==0):
                return -1
                
            #add corresponding affected variables
            for xk in constraints[xi]:
                if(xk==xj):
                    continue
                myq.append((xk,xi))
    
    #print("2")
    #print(domain_values)
    return 1       
    
def dfs_bt_plus(constraints, assignments, domain_vals):
    global total_calls
    total_calls+=1
    #check if assignment is complete
    if(assignments[-1]==n):
        #All Assignments have been done...
        print(assignments)
        print("Done")
        return 1
    
    #variable selection
    next_key = select_unassigned_var(assignments, domain_vals)
    #print(next_key)
    ordered_domain_vals = order_domain_vals(constraints, assignments, next_key, domain_vals)
    #print(ordered_domain_vals)
    #print("Key")
    #print(next_key)
    #print(ordered_domain_vals)
    
    for color_tup in ordered_domain_vals:
        color = color_tup[0]
        
        #create a copy of the domain_values
        dom_copy = copy.deepcopy(domain_vals)
        
        #check for consistency
        fails = 0
        assignments[next_key] = color
        assignments[-1]+=1
        #tempvar = copy.copy(domain_vals[next_key])
        dom_copy[next_key]=[color]
        for j in constraints[next_key]:
            if((assignments[j]==color)):
                fails=1
                break;
        if(fails==1):
            #inconsistent assignment
            assignments[next_key] = -1
            assignments[-1]-=1
            #domain_vals[next_key]=tempvar
            continue
        
        #run the ac3_consistency algo check
        #get the tuples as suggested in the book for only the current variable
        tup_list_next = []
        for nb in constraints[next_key]:
            if(assignments[nb]==-1):
                new_tup = (nb,next_key)
                tup_list_next.append(new_tup)
        
        #create a temporary copy of the domain_vals list
        #temp_list = copy.copy(domain_vals)
        #print("1")
        #print(domain_vals)
        
        
        ac3_res = ac3_consistency(constraints, dom_copy, tup_list_next)
        
        #print("2")
        #print(domain_vals)
        if(ac3_res==1):
            #we'll update the domain values and save a copy for backtracking purposes
            #save_copy_dom_values = copy.copy(domain_vals)
            result = dfs_bt_plus(constraints, assignments, dom_copy)
            if(result!=-1):
                return result
        
        assignments[next_key] = -1
        assignments[-1]-=1
        #domain_vals = temp_list
        #domain_vals[next_key]=tempvar
        
    return -1
    
#returns a -1 in case of an error, 1 in case of success/continuation
def dfs_bt_simple(constraints, assignments):
    global total_calls
    total_calls+=1
    #check if assignment is complete
    if(assignments[-1]==n):
        #All Assignments have been done...
        #print(assignments)
        print("Done")
        return 1
    
    next_key = -1
    #variable selection
    #TODO algo to select best variable
    #print(assignments)
    for i in range(0, n):
        if(assignments[i]==-1):
            next_key = i
            break;
    
    #print("Key ",next_key)
    #iterate over all possible assignments to the variable
    for i in range(0,k):
        
        #check for consistency
        #need to impose stricter checks
        fails = 0
        assignments[next_key] = i
        assignments[-1]+=1
        #print(i)
        #print(assignments)
        #print('Constraints ', constraints[next_key])
        #check the neighbors of the current variable for assignment consistency
        for j in constraints[next_key]:
            if((assignments[j]==assignments[next_key])):
                fails=1
                break;
        if(fails==0):
            
            result = dfs_bt_simple(constraints, assignments)
            if(result == 1):
                return 1
        assignments[-1]-=1
        assignments[next_key] = -1
    return -1
    
#The main function
if __name__ == '__main__':
    if ((len(sys.argv))!=4):
        print("Incorrect Number of arguments")
        sys.exit()
    
    #arguments to the program
    input_file = open(sys.argv[1], 'r')
    output_file = open(sys.argv[2], 'w')
    mode = sys.argv[3]
    
    i = 0
    constraints = defaultdict(list)
    assignments = defaultdict(int)
    domain_vals = []
    initial_constraints = []
    for line in input_file:
        current_line = line.strip().split()
        
        if(i==0):
            n = int(current_line[0])
            m = int(current_line[1])
            k = int(current_line[2])
            i+=1;
        
        else:
            constraints[int(current_line[0])].append(int(current_line[1]))
            constraints[int(current_line[1])].append(int(current_line[0]))
            initial_constraints.append((int(current_line[0]),int(current_line[1])))
    
    #initialize all assignments to -1, signifying that they are not assigned as of now
    #add all possible colors to the domains of all the variables
    for lvar in range(0,n):
        assignments[lvar] = -1
        domain_vals.append(list(range(0,k)))
    
    assignments[-1] = 0
    
    #print(constraints)
    #print(assignments)
    #print(domain_vals)
    global total_calls
    global total_prunes
    #for a simple dfs with backtracking
    if(mode=="0"):
        t1 = time.time()
        total_calls = 0
        result = dfs_bt_simple(constraints, assignments)
        t2 = time.time()
        time_diff = t2-t1
        print(time_diff)
        print(total_calls)
        if (result==-1):
            output_file.write("No Answer")
        else:
            for i in range(0,n):
                output_file.write(str(assignments[i]))
                output_file.write("\n")
    else:
        t1 = time.time()
        total_calls = 0
        total_prunes = 0
        result = dfs_bt_plus(constraints, assignments, domain_vals)
        t2 = time.time()
        time_diff = t2-t1
        print(time_diff)
        print(total_calls)
        print(total_prunes)
        if (result==-1):
            output_file.write("No Answer")
        else:
            for i in range(0,n):
                output_file.write(str(assignments[i]))
                output_file.write("\n")
    print(constraints)
    print(assignments)
    print(domain_vals)