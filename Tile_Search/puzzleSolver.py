import sys
import math
import time
import random
import heapq
from copy import deepcopy
import numpy as np

#The goal states are described as 2d arrays and the blank state contains the value 9(any large number should suffice)
#The maximum valid value in both arrays being less than 9, it is chosen to make the computation a little more convenient
#The goal state for a 3*3 array
goal8 = np.array([[1,2,3],[4,5,6],[7,8,9]])
#The goal state for a 4*4 array
goal16 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
#locations for the elements in a 8 puzzle
loc8 = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
#locations for the elements in a 16 puzzle
loc16 = np.array([[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]])

found = 0
path = []

def move_blank_tile(puz,i,j,direction):
	temp = puz[i][j]
	#move up
	if(direction==1):
		puz[i][j] = puz[i-1][j]
		puz[i-1][j] = temp
	#move down
	if(direction==2):
		puz[i][j] = puz[i+1][j]
		puz[i+1][j] = temp
	#move left
	if(direction==3):
		puz[i][j] = puz[i][j-1]
		puz[i][j-1] = temp
	#move right
	if(direction==4):
		puz[i][j] = puz[i][j+1]
		puz[i][j+1] = temp
		
#object describing the current state of the class
class puzzle8_state:
	
	def __init__(self, arr, gn, h, ival, jval, l, r, u, d, p):
		self.current_state = arr.copy()
		self.g_n = gn
		self.heuristic = 0
		self.f_n = 0
		if (h==1):
			#calculate the number of tiles not in their goal position
			for i in range(0,3):
				for j in range(0,3):
					if ((self.current_state[i][j]!=goal8[i][j]) and (self.current_state[i][j]!=9)):
						self.heuristic+=1
		if (h==2):
			#calculate the manhattan distances
			for i in range(0,3):
				for j in range(0,3):
					if ((self.current_state[i][j]!=goal8[i][j]) and (self.current_state[i][j]!=9)):
						d_ver = abs(loc8[(self.current_state[i][j])-1][0]-i)
						d_hor = abs(loc8[(self.current_state[i][j])-1][1]-j)
						self.heuristic = self.heuristic + d_ver + d_hor
					
		self.f_n = self.g_n + self.heuristic
		self.midi = ival
		self.midj = jval
		self.left = l
		self.right = r
		self.up = u
		self.down = d
		self.path = p
	
	#function to calculate the possible moves
	def possible_moves(self):
		if(self.midi == 0):
			self.up = 0
		if(self.midj == 0):
			self.left = 0
		if(self.midi == 2):
			self.down = 0
		if(self.midj == 2): 
			self.right = 0
			
class puzzle16_state:
	
	def __init__(self, arr, gn, h, ival, jval, l, r, u, d, p):
		self.current_state = arr.copy()
		self.g_n = gn
		self.heuristic = 0
		self.f_n = 0
		if (h==1):
			#calculate the number of tiles not in their goal position
			for i in range(0,4):
				for j in range(0,4):
					if ((self.current_state[i][j]!=goal16[i][j]) and (self.current_state[i][j]!=16)):
						self.heuristic+=1
		if (h==2):
			#calculate the manhattan distances
			for i in range(0,4):
				for j in range(0,4):
					if ((self.current_state[i][j]!=goal16[i][j]) and (self.current_state[i][j]!=16)):
						d_ver = abs(loc16[(self.current_state[i][j])-1][0]-i)
						d_hor = abs(loc16[(self.current_state[i][j])-1][1]-j)
						self.heuristic = self.heuristic + d_ver + d_hor
					
		self.f_n = self.g_n + self.heuristic
		self.midi = ival
		self.midj = jval
		self.left = l
		self.right = r
		self.up = u
		self.down = d
		self.path = p
	
	#function to calculate the possible moves
	def possible_moves(self):
		if(self.midi == 0):
			self.up = 0
		if(self.midj == 0):
			self.left = 0
		if(self.midi == 3):
			self.down = 0
		if(self.midj == 3): 
			self.right = 0

def search(node, gn, threshold, dir, heuristic):
	if(node.f_n > threshold):
		return node.f_n
	if(np.array_equal(node.current_state,goal8)):
		found = 1
		path.append(dir)
		return 0
		
	node.possible_moves()
	#move up
	minval = 9999
	if(node.up==1):
		#check for upper move
		u_arr = node.current_state.copy()
		move_blank_tile(u_arr, node.midi, node.midj, 1)
		node_up = puzzle8_state(u_arr, node.g_n+1, heuristic, node.midi-1, node.midj, 1, 1, 1, 0, 0)
		retval = search(node_up, node_up.g_n, threshold, 'U', heuristic)
		if(retval == 0):
			found = 1
			path.append(dir)
			return 0
		if(retval < minval):
			minval = retval
	if(node.down==1):
		#check for lower move
		d_arr = node.current_state.copy()
		move_blank_tile(d_arr, node.midi, node.midj, 2)
		node_down = puzzle8_state(d_arr, node.g_n+1, heuristic, node.midi+1, node.midj, 1, 1, 0, 1, 0)
		retval = search(node_down, node_down.g_n, threshold, 'D', heuristic)
		if(retval == 0):
			found = 1
			path.append(dir)
			return 0
		if(retval < minval):
			minval = retval
	if(node.left==1):
		#check for left move
		l_arr = node.current_state.copy()
		move_blank_tile(l_arr, node.midi, node.midj, 3)
		node_left = puzzle8_state(l_arr, node.g_n+1, heuristic, node.midi, node.midj-1, 1, 0, 1, 1, 0)
		retval = search(node_left, node_left.g_n, threshold, 'L', heuristic)
		if(retval == 0):
			found = 1
			path.append(dir)
			return 0
		if(retval < minval):
			minval = retval
	if(node.right==1):
		#check for right move
		r_arr = node.current_state.copy()
		move_blank_tile(r_arr, node.midi, node.midj, 4)
		node_right = puzzle8_state(r_arr, node.g_n+1, heuristic, node.midi, node.midj+1, 0, 1, 1, 1, 0)
		retval = search(node_right, node_right.g_n, threshold, 'R', heuristic)
		if(retval == 0):
			found = 1
			path.append(dir)
			return 0
		if(retval < minval):
			minval = retval
	return minval
	
def search16(node, gn, threshold, dir, heuristic):
	if(node.f_n > threshold):
		return node.f_n
	if(np.array_equal(node.current_state,goal16)):
		found = 1
		path.append(dir)
		return 0
		
	node.possible_moves()
	#move up
	minval = 9999
	if(node.up==1):
		#check for upper move
		u_arr = node.current_state.copy()
		move_blank_tile(u_arr, node.midi, node.midj, 1)
		node_up = puzzle16_state(u_arr, node.g_n+1, heuristic, node.midi-1, node.midj, 1, 1, 1, 0, 0)
		retval = search16(node_up, node_up.g_n, threshold, 'U', heuristic)
		if(retval == 0):
			found = 1
			path.append(dir)
			return 0
		if(retval < minval):
			minval = retval
	if(node.down==1):
		#check for lower move
		d_arr = node.current_state.copy()
		move_blank_tile(d_arr, node.midi, node.midj, 2)
		node_down = puzzle16_state(d_arr, node.g_n+1, heuristic, node.midi+1, node.midj, 1, 1, 0, 1, 0)
		retval = search16(node_down, node_down.g_n, threshold, 'D', heuristic)
		if(retval == 0):
			found = 1
			path.append(dir)
			return 0
		if(retval < minval):
			minval = retval
	if(node.left==1):
		#check for left move
		l_arr = node.current_state.copy()
		move_blank_tile(l_arr, node.midi, node.midj, 3)
		node_left = puzzle16_state(l_arr, node.g_n+1, heuristic, node.midi, node.midj-1, 1, 0, 1, 1, 0)
		retval = search16(node_left, node_left.g_n, threshold, 'L', heuristic)
		if(retval == 0):
			found = 1
			path.append(dir)
			return 0
		if(retval < minval):
			minval = retval
	if(node.right==1):
		#check for right move
		r_arr = node.current_state.copy()
		move_blank_tile(r_arr, node.midi, node.midj, 4)
		node_right = puzzle16_state(r_arr, node.g_n+1, heuristic, node.midi, node.midj+1, 0, 1, 1, 1, 0)
		retval = search16(node_right, node_right.g_n, threshold, 'R', heuristic)
		if(retval == 0):
			found = 1
			path.append(dir)
			return 0
		if(retval < minval):
			minval = retval
	return minval

#The main function
if __name__ == '__main__':
	if ((len(sys.argv))<6):
		print("More arguments needed")
		sys.exit()
	
	#a* or ida
	algo = int(sys.argv[1])
	#size of array
	n = int(sys.argv[2])
	heuristic = int(sys.argv[3])
	ip_file = open(sys.argv[4], 'r')
	op_file = open(sys.argv[5], 'w')
	
	start_state8 = np.empty([3,3],dtype=np.uint8)
	start_state16 = np.empty([4,4],dtype=np.uint8)
	blankx = 0
	blanky = 0
	num_lines = 1
	i=0
	print(n)
	if (n == 3):			
		#read the input from the file specified into the start state array
		for line in ip_file:
			if(num_lines>3):
				break
			num_lines+=1
			if(len(line) == 6):
				start_state8[i][0] = int(line[0])
				start_state8[i][1] = int(line[2])
				start_state8[i][2] = int(line[4])
				i+=1
				continue
			if(len(line)==5):
				#blank tile in the first column
				if(line[0]==','):
					blankx = i
					blanky = 0
					start_state8[i][0] = 9
					start_state8[i][1] = int(line[1])
					start_state8[i][2] = int(line[3])
					i+=1
					continue
				
				#blank tile in the last column
				if(line[3]==','):
					blankx = i
					blanky = 2
					start_state8[i][0] = int(line[0])
					start_state8[i][1] = int(line[2])
					start_state8[i][2] = 9
					i+=1
					continue
				
				#blank tile in the middle
				if(line[1]==',' and line[2]==','):
					blankx = i
					blanky = 1
					start_state8[i][0] = int(line[0])
					start_state8[i][1] = 9
					start_state8[i][2] = int(line[3])
					i+=1
					continue
	
		#def __init__(self, arr, gn, h, ival, jval, l, r, u, d):
		
		if(algo==1):
			temp_list = []
			temp_list.append("S")
			ss = puzzle8_state(start_state8, 0, heuristic, blankx, blanky, 1, 1, 1, 1, temp_list.copy())
			ss.possible_moves()
			
			#create an empty priority queue
			priority_q = []
			
			my_counter = 0

			#using heapq for the operations of a priority queue using the value of f(n) as the priority
			heapq.heappush(priority_q, (ss.f_n, my_counter, ss))
			my_counter+=1
			
			#calculate the different states that can be reached from the starting state and insert into the priority queue
			#also making sure that the reverse of the move cannot be inserted into the queue
			
			found = 0
			
			already_visited = [];
			
			#Start iterating through the priority queue until its empty
			while(priority_q):
				#print(len(priority_q))
				#pop the node with the lowest value of f(n)
				cur_node = heapq.heappop(priority_q)
				already_visited.append(cur_node[2].current_state.copy())
				if (np.array_equal(cur_node[2].current_state, goal8)):
					print("Reached goal state")
					print(cur_node[2].path)
					xx=0
					for dir in cur_node[2].path:
						if(xx==0):
							xx=1
							continue
						if(xx==1):
							xx=2
							op_file.write(dir)
							continue
						op_file.write(",")
						op_file.write(dir)
					op_file.close()
					found = 1
					break;
				
				path_so_far = cur_node[2].path
				cost_to_reach = cur_node[2].g_n
				blankx = cur_node[2].midi
				blanky = cur_node[2].midj
				temp_listl = cur_node[2].path.copy()
				temp_listl.append("L")
				temp_listr = cur_node[2].path.copy()
				temp_listr.append("R")
				temp_listu = cur_node[2].path.copy()
				temp_listu.append("U")
				temp_listd = cur_node[2].path.copy()
				temp_listd.append("D")
				
				#self, arr, gn, h1, h2, xpos, ypos, l, r, u, d, p
				#move the blank tile in different directions
				if(cur_node[2].left!=0):
					#print("L")
					sl = puzzle8_state(cur_node[2].current_state.copy(), cost_to_reach+1, heuristic, blankx, blanky-1, 1, 0, 1, 1, temp_listl.copy())
					move_blank_tile(sl.current_state,blankx,blanky,3)
					sl.possible_moves()
					q = 0
					#print(sl.current_state)
					#print(sl.f_n).
					#print("LE")
					for entry in already_visited:

						#print(entry)
						if(np.array_equal(sl.current_state,entry)):
							q=1
							break;
					
					z = 0
					#print("LEE")
					if (q==0):
						for entry in priority_q:
							#print(entry[2].current_state)
							if(np.array_equal(sl.current_state,entry[2].current_state)):
								q=1
								if(sl.f_n < entry[2].f_n):
									q = 0
									del priority_q[z]
									heapq.heapify(priority_q)
									break;
								break
							z+=1
					if (q==0):
						heapq.heappush(priority_q, (sl.f_n, my_counter, sl))
						my_counter+=1
				#try and move the blank tile right
				if(cur_node[2].right!=0):
					#print("R")
					sr = puzzle8_state(cur_node[2].current_state.copy(), cost_to_reach+1, heuristic, blankx, blanky+1, 0, 1, 1, 1, temp_listr.copy())
					move_blank_tile(sr.current_state,blankx,blanky,4)
					sr.possible_moves()
					q = 0
					#print(sr.current_state)
					#print(sr.f_n)
					for entry in already_visited:
						#print(entry[2].current_state)
						if(np.array_equal(sr.current_state,entry)):
							q=1
							break;
					
					z = 0
					if (q==0):
						for entry in priority_q:
							if(np.array_equal(sr.current_state,entry[2].current_state)):
								q=1
								if(sr.f_n < entry[2].f_n):
									q = 0
									del priority_q[z]
									heapq.heapify(priority_q)
									break;
								break
							z+=1
					if (q==0):
						heapq.heappush(priority_q, (sr.f_n, my_counter, sr))
						my_counter+=1
				#try and move the blank tile up
				if(cur_node[2].up!=0):
					#print("U")
					su = puzzle8_state(cur_node[2].current_state.copy(), cost_to_reach+1, heuristic, blankx-1, blanky, 1, 1, 1, 0, temp_listu.copy())
					move_blank_tile(su.current_state,blankx,blanky,1)
					su.possible_moves()
					#print(su.current_state)
					#print(su.f_n)
					q = 0
					for entry in already_visited:
						#print(entry[2].current_state)
						if(np.array_equal(su.current_state,entry)):
							q=1
							break;
					
					z = 0
					if (q==0):
						for entry in priority_q:
							if(np.array_equal(su.current_state,entry[2].current_state)):
								q=1
								if(su.f_n < entry[2].f_n):
									q = 0
									del priority_q[z]
									heapq.heapify(priority_q)
									break;
								break
							z+=1
					if (q==0):
						heapq.heappush(priority_q, (su.f_n, my_counter, su))
						my_counter+=1
				#try and move the blank tile down
				if(cur_node[2].down!=0):
					#print("Down")
					sd = puzzle8_state(cur_node[2].current_state.copy(), cost_to_reach+1, heuristic, blankx+1, blanky, 1, 1, 0, 1, temp_listd.copy())
					move_blank_tile(sd.current_state,blankx,blanky,2)
					sd.possible_moves()
					#print(sd.current_state)
					#print(sd.g_n)
					q = 0
					for entry in already_visited:
						#print(entry[2].current_state)
						if(np.array_equal(sd.current_state,entry)):
							q=1
							break;
					
					z = 0
					if (q==0):
						for entry in priority_q:
							if(np.array_equal(sd.current_state,entry[2].current_state)):
								q=1
								if(sd.f_n < entry[2].f_n):
									q = 0
									del priority_q[z]
									heapq.heapify(priority_q)
									break;
								break
							z+=1
					if (q==0):
						heapq.heappush(priority_q, (sd.f_n, my_counter, sd))
						my_counter+=1
			
			
			sys.exit()
			
		ss = puzzle8_state(start_state8, 0, heuristic, blankx, blanky, 1, 1, 1, 1, 0)
		threshold = ss.f_n
		while (found==0):
			retval = search(ss, 0, threshold, 'S', heuristic)
			if(retval == 0):
				print("Done")
				
				break
			if (retval !=0):
				threshold = retval
				
		print(list(reversed(path)))
		xx=0
		for dir in list(reversed(path)):
			if(xx==0):
				xx=1
				continue
			if(xx==1):
				xx=2
				op_file.write(dir)
				continue
			op_file.write(",")
			op_file.write(dir)
		op_file.close()
		
	if (n == 4):
		#read the input from the file specified into the start state array
		for line in ip_file:
			current_line = line.strip().split(",")
			for t in range(0,4):
				if(current_line[t]==""):
					start_state16[i][t] = 16
					blankx = i
					blanky = t
					continue
				start_state16[i][t] = int(current_line[t])
			i+=1
			if(num_lines>4):
				break
			num_lines+=1
	
		#def __init__(self, arr, gn, h, ival, jval, l, r, u, d):
		
		if(algo==1):
			temp_list = []
			temp_list.append("S")
			ss = puzzle16_state(start_state16, 0, heuristic, blankx, blanky, 1, 1, 1, 1, temp_list.copy())
			ss.possible_moves()
			
			#create an empty priority queue
			priority_q = []
			
			my_counter = 0

			#using heapq for the operations of a priority queue using the value of f(n) as the priority
			heapq.heappush(priority_q, (ss.f_n, my_counter, ss))
			my_counter+=1
			
			#calculate the different states that can be reached from the starting state and insert into the priority queue
			#also making sure that the reverse of the move cannot be inserted into the queue
			
			found = 0
			
			already_visited = [];
			
			#Start iterating through the priority queue until its empty
			while(priority_q):
				#print(len(priority_q))
				#pop the node with the lowest value of f(n)
				cur_node = heapq.heappop(priority_q)
				already_visited.append(cur_node[2].current_state.copy())
				if (np.array_equal(cur_node[2].current_state, goal16)):
					print("Reached goal state")
					print(cur_node[2].path)
					xx=0
					for dir in cur_node[2].path:
						if(xx==0):
							xx=1
							continue
						if(xx==1):
							xx=2
							op_file.write(dir)
							continue
						op_file.write(",")
						op_file.write(dir)
					op_file.close()
					found = 1
					break;
				
				path_so_far = cur_node[2].path
				cost_to_reach = cur_node[2].g_n
				blankx = cur_node[2].midi
				blanky = cur_node[2].midj
				temp_listl = cur_node[2].path.copy()
				temp_listl.append("L")
				temp_listr = cur_node[2].path.copy()
				temp_listr.append("R")
				temp_listu = cur_node[2].path.copy()
				temp_listu.append("U")
				temp_listd = cur_node[2].path.copy()
				temp_listd.append("D")
				
				#self, arr, gn, h1, h2, xpos, ypos, l, r, u, d, p
				#move the blank tile in different directions
				if(cur_node[2].left!=0):
					#print("L")
					sl = puzzle16_state(cur_node[2].current_state.copy(), cost_to_reach+1, heuristic, blankx, blanky-1, 1, 0, 1, 1, temp_listl.copy())
					move_blank_tile(sl.current_state,blankx,blanky,3)
					sl.possible_moves()
					q = 0
					#print(sl.current_state)
					#print(sl.f_n).
					#print("LE")
					for entry in already_visited:

						#print(entry)
						if(np.array_equal(sl.current_state,entry)):
							q=1
							break;
					
					z = 0
					#print("LEE")
					if (q==0):
						for entry in priority_q:
							#print(entry[2].current_state)
							if(np.array_equal(sl.current_state,entry[2].current_state)):
								q=1
								if(sl.f_n < entry[2].f_n):
									q = 0
									del priority_q[z]
									heapq.heapify(priority_q)
									break;
								break
							z+=1
					if (q==0):
						heapq.heappush(priority_q, (sl.f_n, my_counter, sl))
						my_counter+=1
				#try and move the blank tile right
				if(cur_node[2].right!=0):
					#print("R")
					sr = puzzle16_state(cur_node[2].current_state.copy(), cost_to_reach+1, heuristic, blankx, blanky+1, 0, 1, 1, 1, temp_listr.copy())
					move_blank_tile(sr.current_state,blankx,blanky,4)
					sr.possible_moves()
					q = 0
					#print(sr.current_state)
					#print(sr.f_n)
					for entry in already_visited:
						#print(entry[2].current_state)
						if(np.array_equal(sr.current_state,entry)):
							q=1
							break;
					
					z = 0
					if (q==0):
						for entry in priority_q:
							if(np.array_equal(sr.current_state,entry[2].current_state)):
								q=1
								if(sr.f_n < entry[2].f_n):
									q = 0
									del priority_q[z]
									heapq.heapify(priority_q)
									break;
								break
							z+=1
					if (q==0):
						heapq.heappush(priority_q, (sr.f_n, my_counter, sr))
						my_counter+=1
				#try and move the blank tile up
				if(cur_node[2].up!=0):
					#print("U")
					su = puzzle16_state(cur_node[2].current_state.copy(), cost_to_reach+1, heuristic, blankx-1, blanky, 1, 1, 1, 0, temp_listu.copy())
					move_blank_tile(su.current_state,blankx,blanky,1)
					su.possible_moves()
					#print(su.current_state)
					#print(su.f_n)
					q = 0
					for entry in already_visited:
						#print(entry[2].current_state)
						if(np.array_equal(su.current_state,entry)):
							q=1
							break;
					
					z = 0
					if (q==0):
						for entry in priority_q:
							if(np.array_equal(su.current_state,entry[2].current_state)):
								q=1
								if(su.f_n < entry[2].f_n):
									q = 0
									del priority_q[z]
									heapq.heapify(priority_q)
									break;
								break
							z+=1
					if (q==0):
						heapq.heappush(priority_q, (su.f_n, my_counter, su))
						my_counter+=1
				#try and move the blank tile down
				if(cur_node[2].down!=0):
					#print("Down")
					sd = puzzle16_state(cur_node[2].current_state.copy(), cost_to_reach+1, heuristic, blankx+1, blanky, 1, 1, 0, 1, temp_listd.copy())
					move_blank_tile(sd.current_state,blankx,blanky,2)
					sd.possible_moves()
					#print(sd.current_state)
					#print(sd.g_n)
					q = 0
					for entry in already_visited:
						#print(entry[2].current_state)
						if(np.array_equal(sd.current_state,entry)):
							q=1
							break;
					
					z = 0
					if (q==0):
						for entry in priority_q:
							if(np.array_equal(sd.current_state,entry[2].current_state)):
								q=1
								if(sd.f_n < entry[2].f_n):
									q = 0
									del priority_q[z]
									heapq.heapify(priority_q)
									break;
								break
							z+=1
					if (q==0):
						heapq.heappush(priority_q, (sd.f_n, my_counter, sd))
						my_counter+=1
			
			
			sys.exit()

		ss = puzzle16_state(start_state16, 0, heuristic, blankx, blanky, 1, 1, 1, 1, 0)
		threshold = ss.f_n
		while (found==0):
			retval = search16(ss, 0, threshold, 'S', heuristic)
			if(retval == 0):
				print("Done")
				
				break
			if (retval !=0):
				threshold = retval
				
		print(list(reversed(path)))
		xx=0
		for dir in list(reversed(path)):
			if(xx==0):
				xx=1
				continue
			if(xx==1):
				xx=2
				op_file.write(dir)
				continue
			op_file.write(",")
			op_file.write(dir)
		op_file.close()