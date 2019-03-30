import sys
import math
import time
import random
import heapq
from copy import deepcopy
import numpy as np
from collections import defaultdict
import random
from random import randint

n=0
m=0
k=0

#The main function
if __name__ == '__main__':
    if ((len(sys.argv))!=3):
        print("Incorrect Number of arguments")
        sys.exit()
    
    #arguments to the program
    input_file = open(sys.argv[1], 'r')
    output_file = open(sys.argv[2], 'w')
    
    i = 0
    constraints = defaultdict(list)
    assignments = defaultdict(int)
    
    #read inputs
    for line in input_file:
        current_line = line.strip().split()
        
        if(i==0):
            n = int(current_line[0])
            m = int(current_line[1])
            k = int(current_line[2])
            i=1;
        
        else:
            constraints[int(current_line[0])].append(int(current_line[1]))
            constraints[int(current_line[1])].append(int(current_line[0]))
    
    #randomly assign values to each variable
    for lvar in range(0,n):
        assignments[lvar] = randint(0,k-1)
        
    #print(assignments)
    soln_found = 0
    #min conflicts start
    #max value is 250000
    t1 = time.time()
    #get the number of steps taken
    num_steps = 0
    for i in range(0,250000):
        num_steps+=1
        #empty list to store conflicted variables
        conf_list = []
        #variable to see if we have a solution or conflict
        conf = 0
        #check for possiblity of a solution given the current assignments
        for j in range(0,n):
            for q in constraints[j]:
                if(assignments[j]==assignments[q]):
                    conf+=1
                    conf_list.append(j)
                    break
        
        #total number of conflicting variables at this point is conf
        #ie. all variables that have a conflict corresponding to the list of constraints they have
        
        if(conf==0):
            print("Solution found...")
            soln_found = 1
            break;                    
        
        #select a random conflicted variable from the list created
        cur_var = random.choice(conf_list)
        
        #get a random value between 1 and 7
        #if the value is less than 3, then just assign a random color to the variable
        #else, we continue to find the color which results in minimum number of conflicts
        p = randint(1,7)
        if(p<3):
            assignments[cur_var] = randint(0,k-1)
            continue
        
        min_conflicts = 9999
        #select value for the random variable that minimizes conflicts
        for j in range(0,k):
            #check if the color is the current color of the variable
            if(j==assignments[cur_var]):
                continue
            #assignments[cur_var] = j
            
            num_conf=0
            #check for conflicts now that we have an assignment for the chosen variable
            for q in constraints[cur_var]:
                if(assignments[q]==j):
                    num_conf+=1
            
            #now we have the number of conflicts corresponding to an assignment 'j' for the chosen variable
            if(num_conf<min_conflicts):
                min_conflicts = num_conf
                color_chosen = j

        #now that we have selected the value which minimizes conflicts (the first such value is used)
        assignments[cur_var] = color_chosen
        
    t2 = time.time()
    time_diff = t2-t1
    print(time_diff)
    print(num_steps)
         
    #print(assignments)
    #print(constraints)
    if(soln_found==0):
        print("No Answer")
        output_file.write("No Answer")
    else:
        print("Answer found")
        for val in range(0,n):
            output_file.write(str(assignments[val]))
            output_file.write("\n")