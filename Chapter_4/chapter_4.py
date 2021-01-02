from numba import cuda, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import time
from math import log, ceil
from scipy import stats
from matplotlib import pyplot as plt
import argparse

@cuda.jit(device=True)
def packet(rng_states,thread_id,q):
	x=0
	y=False
	while y==False:
		y=(xoroshiro128p_uniform_float32(rng_states, thread_id)>q)
		x+=1
	return x

@cuda.jit
def fountain(rng_states, k_max, rvspacef, outf, outcf, outif, q):
	thread_id = cuda.grid(1)
	rvspacef[thread_id][1][0] = packet(rng_states,thread_id,q)
	rvspacef[thread_id][0][0] = max(rvspacef[thread_id][0][0],rvspacef[thread_id][1][0])
	for k in range(1,k_max):
		rvspacef[thread_id][1][k] = rvspacef[thread_id][1][k-1]+ packet(rng_states,thread_id,q)
		rvspacef[thread_id][0][k] = max(rvspacef[thread_id][0][k],rvspacef[thread_id][1][k])
	cuda.syncthreads()

@cuda.jit
def fountaindat(rng_states, k_max, rvspacef, outf, outcf, outif, q):
	thread_id = cuda.grid(1)
	cuda.syncthreads()
	if thread_id<k_max:
		for j in range(1000):
			outcf[thread_id][j] = 0
			outf[thread_id][j] = 0
		for i in range(threads_per_block*blocks):
			outf[thread_id][rvspacef[i][0][thread_id]-(thread_id+1)] += 1
		outcf[thread_id][0] = outf[thread_id][0]
		outif[thread_id]=0
		for j in range(1,1000):
			outcf[thread_id][j] = outcf[thread_id][j-1] + outf[thread_id][j]
		for j in range(999):
			if outcf[thread_id][j] <0.9*blocks*threads_per_block:
				outif[thread_id]+=1

@cuda.jit
def arq(rng_states, n_max, rvspacea, outa, k, q):
	thread_id = cuda.grid(1)
	rvspacea[thread_id][1][0] = packet(rng_states,thread_id,q)
	rvspacea[thread_id][0][0] += rvspacea[thread_id][1][0]
	for n in range(1,n_max):
		rvspacea[thread_id][1][n] = max(rvspacea[thread_id][1][n-1], packet(rng_states,thread_id,q))
		rvspacea[thread_id][0][n] += rvspacea[thread_id][1][n]
	cuda.syncthreads()
	if thread_id<n_max:
		for j in range(4000):
			outa[thread_id][j] = 0
		for i in range(threads_per_block*blocks):
			if rvspacea[i][0][thread_id]-(k+1) < 0:
				cuda.syncthreads()
				if rvspacea[i][0][thread_id]-(k+1) < 0:
					print("rvspacea[i][0][thread_id]-(k+1)")
			if rvspacea[i][0][thread_id]-(k+1) > 4999:
				print(k)
			outa[thread_id][rvspacea[i][0][thread_id]-(k+1)] += 1
	
@cuda.jit
def nplot(sims, simsa, k, n_max, out, outa):
	thread_id=cuda.grid(1)
	if thread_id<n_max:
		out[thread_id]=0
		for i in range(0, 1000):
			out[thread_id] += sims[thread_id][i]*(i+k)
	if thread_id>n_max-1 and thread_id<2*n_max:
		outa[thread_id-n_max]=0
		for i in range(0, 4000):
			outa[thread_id-n_max] += simsa[thread_id-n_max][i]*(i+k)

@cuda.jit
def kplot(sims,simsa, k_max, n, out, outa):
	thread_id=cuda.grid(1)
	if thread_id<k_max:
		out[thread_id]=0
		for i in range(0, 1000):
			out[thread_id] += sims[thread_id][i]*(thread_id +1+i)
	if thread_id>k_max-1 and thread_id<2*k_max:
		outa[thread_id-k_max]=0
		for i in range(0, 4000):
			outa[thread_id-k_max] += simsa[thread_id-k_max][i]*((thread_id-k_max)+1+i)

def f(beta, alpha, q):
	if alpha==0:
		return -beta*log(q)
	else:
		return beta* (((alpha/beta)*log(alpha/(beta*(1-q)))) + ((1-(alpha/beta))*log((1-(alpha/beta))/q)))

def betasolve(alpha, q):
	step=0.1
	b1=alpha/(1-q)
	b2=b1+0.1
	while f(b2, alpha, q)<1:
		b1=b2	
		step*=2	
		b2+=step
	while step>10**(-15):
		step*=0.5
		if f(b1+step, alpha, q)<=1:
			b1+=step
		else:
			b2-=step
	return b2

parser=argparse.ArgumentParser(description="Simulation demonstrating performance of a fountain code against ARQ.")
subparsers = parser.add_subparsers(help='Run simulations (CUDA required), else just run graphs (CUDA not required). Use sims -h to show further command line options.', dest='mode')

sims_parser = subparsers.add_parser("sims")

sims_parser.add_argument("sims", type=int, help="Sims multiplier (sims/3072).")
sims_parser.add_argument("q1", type=float, help="Minimum channel erasure probability.")
sims_parser.add_argument("q2", type=float, help="Maximum channel erasure probability.")
sims_parser.add_argument("qsteps", type=int, help="Number of steps in channel erasure probability.")
sims_parser.add_argument("n", type=int, help="Maximum number of receivers.")
sims_parser.add_argument("k", type=int, help="Maximum message length (packets).")
args=parser.parse_args()

threads_per_block = 128
blocks_multiplier = 24

if args.mode==None:
	print("Loading data, please wait")
	sims,q1,q2,qsteps,n,k = np.load("data/simsqq.npy")
	n=int(n)
	k=int(k)
	qsteps=int(qsteps)
	simsf=np.zeros((qsteps, n, k, 1000),dtype=np.int32)
	simscf=np.zeros((qsteps, n, k, 1000),dtype=np.int32)
	simsif=np.zeros((qsteps, n, k), dtype=np.int32)
	simsa=np.zeros((qsteps, k, n, 4000),dtype=np.int32)
	sums=np.zeros((qsteps, k,n), dtype=np.int32)
	sumsarq=np.zeros((qsteps, k,n), dtype=np.int32)
	sumsk=np.zeros((qsteps, n,k), dtype=np.int32)
	sumsarqk=np.zeros((qsteps, n,k), dtype=np.int32)
	for i in range(qsteps):
		q=q1+((i/(qsteps-1))*(q2-q1))
		sums[i]=np.load("data/"+str(q)+"processed1.npy")
		sumsarq[i]=np.load("data/"+str(q)+"processed2.npy")
		sumsk[i]=np.load("data/"+str(q)+"processed3.npy")
		sumsarqk[i]=np.load("data/"+str(q)+"processed4.npy")
	print("Data loaded.")
	blocks=blocks_multiplier*sims
	print("n="+str(n)+", k="+str(k)+", q="+str(q1)+"..."+str(q2)+", "+str(int(blocks*threads_per_block))+" simulations.")

	beta=np.zeros((qsteps,101),dtype=np.float64)
	alpha=np.array(range(0,101))
	alpha2 = alpha *0.1
	fig2=plt.figure()
	for i in range(qsteps):
		q=q1+((i/(qsteps-1))*(q2-q1))
		for a in alpha:
			beta[i][a]=betasolve(a/10,q)
		plt.plot(np.divide(alpha2,(1-q)*beta[i]),beta[i], label="Fountain code (q="+str(q)+")")
	plt.ylabel("Delay/log(n)")
	plt.xlabel("Throughput/Capacity")
	plt.legend()
	plt.show()
	
	colours=['blue', 'darkorange', 'darkgreen','c','m','y','b','g','r','k']
	for i in range(qsteps):
		q=q1+((i/(qsteps-1))*(q2-q1))
		plt.plot(alpha2, beta[i], label="beta_alpha (q="+str(q)+")", color=colours[i])
		plt.plot(alpha2, (alpha2 + np.sqrt(2*q*alpha2))/(1-q), label="gamma_alpha (q="+str(q)+")", linestyle='--',color=colours[i])
	plt.ylabel("Delay/log(n)")
	plt.xlabel("alpha")
	plt.legend()
	plt.show()

	#plot of transmissions vs k
	fig2=plt.figure()
	for i in range(qsteps):
		q=q1+((i/(qsteps-1))*(q2-q1))
		plt.plot(range(1,k+1),(sumsk[i][n-1]/(blocks*threads_per_block))-((np.array(range(1,k+1)))/(1-q)), label="Fountain code (q="+str(q)+")", color=colours[i])
		plt.plot(range(1,k+1),(sumsarqk[i][n-1]/(blocks*threads_per_block))-((np.array(range(1,k+1)))/(1-q)), label="ARQ (q="+str(q)+")", linestyle='--',color=colours[i])
	plt.ylabel("Excess latency")
	plt.xlabel("k")
	plt.legend()
	plt.show()

	fig2=plt.figure()
	ka=np.array(range(1,k+1))
	alpha=ka/log(20)
	beta=np.zeros(k)
	q=q1+((1/(qsteps-1))*(q2-q1))
	for i in ka:
		beta[i-1]=betasolve(alpha[i-1],q)
	beta*=log(20)
	plt.plot(ka,(sumsk[1][19]/(blocks*threads_per_block))-((np.array(range(1,k+1)))/(1-q)), label="Fountain code (q="+str(q)+", n=20)")
	plt.plot(ka,(ka+np.sqrt(2*q*log(20)*ka))/(1-q) -((np.array(range(1,k+1)))/(1-q)), label="gamma_alpha (q="+str(q)+", n=20)")
	plt.plot(ka,beta -((np.array(range(1,k+1)))/(1-q)), label="beta_alpha (q="+str(q)+", n=20)")

	plt.ylabel("Excess latency")
	plt.xlabel("k")
	plt.legend()
	plt.show()


elif args.mode=="sims":
	np.save("data/simsqq.npy", np.array([args.sims,args.q1,args.q2,args.qsteps,args.n,args.k]))
	blocks=blocks_multiplier * args.sims
	rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
	for q in np.linspace(args.q1, args.q2, args.qsteps):
		print("Simulating q = "	+str(q))
		outf = cuda.to_device(np.zeros((args.k, 1000), dtype=np.int32))
		outcf = cuda.to_device(np.zeros((args.k, 1000), dtype=np.int32))
		outif = cuda.to_device(np.zeros((args.k), dtype=np.int32))
		rvspacef=cuda.to_device(np.zeros((threads_per_block*blocks,2,args.k),dtype=np.int32))
		simsf=np.zeros((args.n, args.k, 1000),dtype=np.int32)
		simscf=np.zeros((args.n, args.k, 1000),dtype=np.int32)
		simsif=np.zeros((args.n, args.k), dtype=np.int32)
		start_time=time.clock()
		for i in range(args.n):
			fountain[blocks, threads_per_block](rng_states, args.k, rvspacef, outf, outcf, outif, q)
			fountaindat[blocks, threads_per_block](rng_states, args.k, rvspacef, outf, outcf, outif, q)
			simsf[i]=outf.copy_to_host()
			simscf[i]=outcf.copy_to_host()
			simsif[i]=outif.copy_to_host()
			print("Fountain: n=", end='')
			print(i+1, end='\r')
		print("Run time (fountain): " + str((time.clock() - start_time)) +" seconds")
		outa = cuda.to_device(np.zeros((args.n, 4000), dtype=np.int32))
		rvspacea=cuda.to_device(np.zeros((threads_per_block*blocks,2,args.n),dtype=np.int32))
		simsa=np.zeros((args.k, args.n, 4000),dtype=np.int32)
		start_time=time.clock()
		for k in range(args.k):
			arq[blocks, threads_per_block](rng_states, args.n, rvspacea, outa, k, q)
			simsa[k]=outa.copy_to_host()
			print("ARQ: k=", end='')
			print(k+1, end='\r')
		print("Run time (ARQ): " + str((time.clock() - start_time)) +" seconds")
	
		start_time=time.clock()
		outf=cuda.to_device(np.zeros(args.n))
		outa=cuda.to_device(np.zeros(args.n))
		sums=np.zeros((args.k,args.n), dtype=np.int32)
		sumsarq=np.zeros((args.k,args.n), dtype=np.int32)
		for i in range(args.k):
			nplot[blocks, threads_per_block](np.ascontiguousarray(simsf.transpose((1,0,2))[i]), simsa[i], i+1, args.n, outf, outa)
			sums[i]=outf.copy_to_host()
			sumsarq[i]=outa.copy_to_host()
			print("Data manipulation (graphs): k=", end='')
			print(i+1, end='\r')
		print("Run time (data manipulation): " + str((time.clock() - start_time)) +" seconds")
	
		start_time=time.clock()
		outf=cuda.to_device(np.zeros(args.k))
		outa=cuda.to_device(np.zeros(args.k))
		sumsk=np.zeros((args.n,args.k), dtype=np.int32)
		sumsarqk=np.zeros((args.n,args.k), dtype=np.int32)
		for i in range(args.n):
			kplot[blocks, threads_per_block](simsf[i], np.ascontiguousarray(simsa.transpose((1,0,2))[i]), args.k, i+1, outf, outa)
			sumsk[i]=outf.copy_to_host()
			sumsarqk[i]=outa.copy_to_host()
			print("Data manipulation (graphs): n=", end='')
			print(i+1, end='\r')
		print("Run time (data manipulation): " + str((time.clock() - start_time)) +" seconds")
		print("Saving data, please wait.")
		np.save("data/"+str(q)+"raw1.npy", simsf)
		np.save("data/"+str(q)+"raw2.npy", simscf)
		np.save("data/"+str(q)+"raw3.npy", simsif)
		np.save("data/"+str(q)+"raw4.npy", simsa)
		np.save("data/"+str(q)+"processed1.npy", np.array(sums))
		np.save("data/"+str(q)+"processed2.npy", np.array(sumsarq))
		np.save("data/"+str(q)+"processed3.npy", np.array(sumsk))
		np.save("data/"+str(q)+"processed4.npy", np.array(sumsarqk))
		print("Data saved successfully.")
