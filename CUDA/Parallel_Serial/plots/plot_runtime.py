import numpy as np
import matplotlib.pyplot as plt

def fread(fname):
	f=open(fname,"r")
	lines=f.readlines()
	x=[]
	y=[]
	for line in lines:
		line=line.split()
		x.append(float(line[0]))
		y.append(float(line[1]))
	return[x,y]


plot_parallel=fread("ParallelRuntime.txt")
plot_serial=fread("SerialRuntime.txt")

plt.title("Size of Input v/s Time_Taken")
plt.xlabel("Size_Of_Input")
plt.ylabel("Time_taken_in_seconds")

plt.plot(plot_serial[0],plot_serial[1])
plt.plot(plot_parallel[0],plot_parallel[1])

plt.legend(["Serial","Parallel"])
plt.show()
