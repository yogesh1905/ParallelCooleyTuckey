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


t1plot=fread("t1.txt")
t2plot=fread("t2.txt")
t4plot=fread("t4.txt")
t8plot=fread("t8.txt")
t16plot=fread("t16.txt")
t32plot=fread("t32.txt")
t64plot=fread("t64.txt")
t128plot=fread("t128.txt")
t256plot=fread("t256.txt")
t512plot=fread("t512.txt")
t1024plot=fread("t1024.txt")

plt.title("Balance v/s Time Taken for different threads for size=2^24")
plt.xlabel("Balance")
plt.ylabel("Time taken in seconds")

plt.plot(t1plot[0],t1plot[1])
plt.plot(t2plot[0],t2plot[1])
plt.plot(t4plot[0],t4plot[1])
plt.plot(t8plot[0],t8plot[1])
plt.plot(t16plot[0],t16plot[1])
plt.plot(t32plot[0],t32plot[1])
plt.plot(t64plot[0],t64plot[1])
plt.plot(t128plot[0],t128plot[1])
plt.plot(t256plot[0],t256plot[1])
plt.plot(t512plot[0],t512plot[1])
plt.plot(t1024plot[0],t1024plot[1])

plt.legend(["1 Thread","2 Threads","4 Threads","8 Threads","16 Threads","32 Threads","64 Threads","128 Threads","256 Threads","512 Threads","1024 Threads"])

plt.show()
