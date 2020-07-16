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


plot_speedup=fread("Speedup.txt")
# plot_serial=fread("Serial.txt")

plt.title("Speedup v/s Size of Input")
plt.xlabel("Size_Of_Input_in_powers_of_2")
plt.ylabel("Speedup")

plt.plot(plot_speedup[0],plot_speedup[1])

plt.show()
