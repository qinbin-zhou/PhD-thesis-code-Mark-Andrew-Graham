#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cmath>
#include <unistd.h>
#include <sys/time.h>
#include "bit-ops.h"
#include <omp.h>

/* Simulation of a 'random linear network coding' approach to allcast, in which
device has a single packet to broadcast to all others, and achieves this by
broadcasting a random linear combination of packets from its buffer at each time
step, over a fixed random graph.*/

/*####################WARNING####################
  WILL ONLY WORK IF sizeof(int)=sizeof(unsigned int)=4. */

#define check(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

/*#################################################################################################
These CUDA kernels are for the making the graph
##################################################################################################*/
__global__ void device_init(curandStateMRG32k3a_t *states, int seed)
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);
	curandStateMRG32k3a_t *state = states + thread_id;
	curand_init(seed, thread_id, 0, state);
}
/*
__global__ void pack_init(curandStateMRG32k3a_t *packstates, int seed)
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);
	curandStateMRG32k3a_t *state = packstates + thread_id;
	curand_init(seed, thread_id, 0, state);
}*/

__global__ void graphgen(curandStateMRG32k3a_t *states, unsigned int *adj, unsigned int *adj2, int n, float p, int notfixed)
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);	
	curandStateMRG32k3a_t state = states[thread_id];
	unsigned int temp = 0;
	unsigned int bit = 1;
	for (int i=0; i<(sizeof(unsigned int)*8); i++){
		temp |= (bit * (curand_uniform(&state)<=p));
		bit <<= 1;
	}
	adj[thread_id] = temp;
	if(notfixed){
		adj2[thread_id] = temp;
	}
	states[thread_id] = state;
}

__global__ void graphthin(curandStateMRG32k3a_t *states, unsigned int *adj, unsigned int *adj2, int n, float p, int notfixed)
{//This just thins the original graph, and not the evolved one.
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);	
	curandStateMRG32k3a_t state = states[thread_id];
	unsigned int temp;
	temp = adj[thread_id];
	unsigned int bit = 1;
	for (int i=0; i<(sizeof(unsigned int)*8); i++){
		temp &= ~(bit * (curand_uniform(&state)>=p));
		bit <<= 1;
	}
	adj[thread_id] = temp;
	if(notfixed){
		adj2[thread_id] = temp;
	}
	states[thread_id] = state;
}

__global__ void graphevo(curandStateMRG32k3a_t *states, unsigned int *adj2, int n, float alpha, float beta)
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);	
	curandStateMRG32k3a_t state = states[thread_id];
	float randf;
	unsigned int temp, temp2;
	unsigned int bit = 1;
	temp = adj2[thread_id];
	for (int i=0; i<(sizeof(unsigned int)*8); i++){
		randf = curand_uniform(&state);
		if (bit & temp){		//There's some warp divergence here, but you'd have to do some
			temp2 = (randf <= beta); // trick with flops otherwise which'd end up
		}else{							// being more expensive.
			temp2 = (randf <= alpha);
		}
		temp |= bit * temp2;
		bit <<= 1;
	}
	adj2[thread_id] = temp;
	states[thread_id] = state;
}

/*#################################################################################################
These CUDA kernels are for the baseline
##################################################################################################*/

__global__ void rxinit(unsigned int *rxed, unsigned int *rxed1, int *sums, int n, float p)
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);	
	for (int i=0; i<n; i++){
		rxed[(i*(n/(sizeof(int)*8)))+thread_id] = 0;
		rxed1[(i*(n/(sizeof(int)*8)))+thread_id] = 0;
	}
	for(int i=0; i<(sizeof(int)*8); i++){
		sums[(thread_id*(sizeof(int)*8))+i]=1;
		SetBit(rxed[thread_id + (((thread_id*sizeof(int)*8)+i)*(n/(sizeof(int)*8)))], i);
		SetBit(rxed1[thread_id + (((thread_id*sizeof(int)*8)+i)*(n/(sizeof(int)*8)))], i);
	}
}
__global__ void choosep(curandStateMRG32k3a_t *states, int *sums, unsigned int *rxed, unsigned int *chosen,  int n, float p)
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);
	curandStateMRG32k3a_t state = states[thread_id];
	int xnum[sizeof(int)*8]={0};
	int xpack[sizeof(int)*8]={0};
	int temprand;
	for (int j=0; j<(sizeof(int)*8); j++){
		temprand = sums[(thread_id*(sizeof(int)*8))+j];
		xnum[j]= temprand * curand_uniform(&state);
		while (temprand==xnum[j]){
			xnum[j]= temprand * curand_uniform(&state);
		}
		xnum[j]++;
	}
	states[thread_id] = state;
	int done=1;
	int bit=1;
	for(int i=0; done==1; i++){
		done=0;
		bit=1;
		temprand = rxed[((i*n)/(sizeof(int)*8)) +thread_id];
		for (int j=0; j<(sizeof(int)*8); j++){
			if(xpack[j]<xnum[j]){
				xpack[j] += ((temprand & bit) != 0);
				if(xpack[j]==xnum[j]){
					chosen[((sizeof(int)*8)*thread_id)+j]=i;

				} else{
					done=1;
				}
			}
			bit <<= 1;
		}
	}
}

__global__ void baseline_xmit(unsigned int *adj, unsigned int *rxed, unsigned int *chosen,  int n, float p)
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);
	for(int i=0; i<n; i++){
		rxed[( (n/(sizeof(int)*8)) * chosen[i]) + thread_id] |= adj[(i * (n/(sizeof(int)*8)) ) + thread_id];
	}
}

__global__ void upsums(unsigned int *rxed, int *sums, int *sums2, int n)
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);
	int sumc[sizeof(int)*8]={0};
	unsigned int bit;
	for (int i=0; i<n; i++){
		bit=1;
		for (int j=0; j<(sizeof(int)*8); j++){
			sumc[j] += ((rxed[(i*(n/(sizeof(int)*8)))+thread_id] & bit)!=0);
			bit <<= 1;
		}
	}
	for (int i=0; i<(sizeof(int)*8); i++){
		sums[(thread_id*(sizeof(int)*8)) + i]=sumc[i];
	}
	if(sums != sums2){
		for (int i=0; i<(sizeof(int)*8); i++){
			sums2[(thread_id*(sizeof(int)*8)) + i]=sumc[i];
		}
	}
}


/*##################################################################################################
  Everything in here is for the baseline for small n.
##################################################################################################*/
__global__ void small_init(unsigned int *rxed, unsigned int *rxed1, int *sums, int n, float p)
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);	
	sums[thread_id]=1;
	for (int i=0; i<n; i++){
		rxed[i+(thread_id*n)] = 0;
		rxed1[i+(thread_id*n)] = 0;
	}
	rxed[thread_id + (n*thread_id)]=1;
	rxed1[thread_id + (n*thread_id)]=1;
}

__global__ void small_choosep(curandStateMRG32k3a_t *states, int *sums, unsigned int *rxed, unsigned int *chosen,  int n, float p)
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);
	int xnum=sums[thread_id]*curand_uniform(states+thread_id);
	while (xnum==sums[thread_id]){
		xnum=sums[thread_id]*curand_uniform(states+thread_id);
	}
	xnum ++;
	int xpack=-1;
	for(int i=0; i<xnum; xpack++){
		i+=rxed[xpack+1+(n*thread_id)];
	}
	chosen[thread_id] = xpack;
}

__global__ void small_xmit(unsigned int *adj, unsigned int *rxed, unsigned int *chosen,  int n, float p)
{
	unsigned int adjcache, bit;
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);
	for(int i=0; i<n/32; i++){
		bit = 1;
		adjcache = adj[i + ((n/32)*thread_id)];
		for(int j=0; j<32; j++){
			rxed[chosen[(32*i)+j] + (n*thread_id)] |= ((adjcache & bit)!= 0); //Beware as this is treating adj as the transpose of the matrix used everywhere else in this program.
			bit <<= 1;
		}
	}
}

__global__ void small_upsums(unsigned int *rxed, int *sums, int *sums2, int n)
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);
	sums[thread_id]=rxed[n*thread_id];
	if(sums != sums2){
		sums2[thread_id]=rxed[n*thread_id];
	}
	for (int i=1; i<n; i++){
		sums[thread_id]+=rxed[i+(n*thread_id)];
		if(sums != sums2){
			sums2[thread_id]+=rxed[i+(n*thread_id)];
		}
	}
}


/*#################################################################################################
These CUDA kernels are for the network coding solution
##################################################################################################*/
__global__ void newsums(unsigned int *adj, int *sums, int *sums2, int n) //Count the number of in-neighbours
{
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);	
	unsigned int bit;
	int temp=0;
	unsigned int adjcache;

	for(int i=0; i<n/(8*sizeof(unsigned int)); i++){
		bit = 1;
		adjcache = adj[((n/(sizeof(unsigned int)*8))*thread_id)+i];
		for(int j=0; j<(sizeof(unsigned int)*8); j++){
			temp += (0 < (bit & adjcache));
			bit <<= 1;
		}
	}
	sums[thread_id]=temp;
	if(sums != sums2){
		sums2[thread_id]=temp;
	}
}

__global__ void init(unsigned int *adj, unsigned int *adj2 ,int *sums, int *sums2, unsigned int *rxed, unsigned int *chosen, unsigned int *solved, int *linsums, int n, int notfixed){
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);
	SetBit(adj[(thread_id+(n*thread_id))/(8*sizeof(unsigned int))], thread_id % (8*sizeof(unsigned int)));
	if(notfixed){
		SetBit(adj2[(thread_id+(n*thread_id))/(8*sizeof(unsigned int))], thread_id % (8*sizeof(unsigned int)));
	}
	for (int i=0; i<(n/(8*sizeof(unsigned int))); i++){
		rxed[(thread_id*(n/(sizeof(unsigned int)*8)))+i] = 0;
		solved[(thread_id*(n/(sizeof(unsigned int)*8)))+i] = 0;
		chosen[(thread_id*(n/(sizeof(unsigned int)*8)))+i] = 0;
	}
	sums[thread_id]=0;
	sums2[thread_id]=0;
	linsums[thread_id]=1;
	SetBit(chosen[((thread_id+(n*thread_id))/(8*sizeof(unsigned int)))], thread_id % (8*sizeof(unsigned int)));
	SetBit(solved[((thread_id+(n*thread_id))/(8*sizeof(unsigned int)))], thread_id % (8*sizeof(unsigned int)));
}

__global__ void makep(curandStateMRG32k3a_t *states, int *sums, unsigned int *rxed, unsigned int *chosen, int n, float sp){
	extern __shared__ unsigned int shared[];
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);	

	curandStateMRG32k3a_t state = states[thread_id];	//TODO:If this section is too slow, you might want to store these things in bits and cache them. You will saturate L1 cache at n=**12
	unsigned int acc = 0; 
	int breaker = 0;
	while (breaker < sums[blockIdx.x]){
		for (int i = breaker + blockDim.x + threadIdx.x; (i< blockDim.x + breaker + (2*n)) && (i < blockDim.x + sums[blockIdx.x])  ; i+=blockDim.x){
			shared[i - breaker] = (curand_uniform(&state)<sp);
		}
		__syncthreads();

		for (int line = breaker + (threadIdx.x % 8); (line<(breaker + (2*n))) && (line < sums[blockIdx.x]); line +=8){	
			acc ^= rxed[((line * n )* (n/(sizeof(unsigned int)*8)) ) + (blockIdx.x * (n/(sizeof(unsigned int)*8)) ) + (threadIdx.x / 8)] * shared[blockDim.x + line - breaker]; 
		}
		__syncthreads();

		breaker += 2*n;
	}
	states[thread_id] = state;
	shared[threadIdx.x] = acc;
	__syncthreads();

	if(threadIdx.x % 2 ==0){
		shared[threadIdx.x] ^= shared[threadIdx.x + 1];
	}
	__syncthreads();
	
	if(threadIdx.x % 4 ==0){
		shared[threadIdx.x] ^= shared[threadIdx.x + 2];
	}
	__syncthreads();


	if(threadIdx.x % 8 ==0){
		chosen[(blockIdx.x * (n/(sizeof(unsigned int)*8)) ) + (threadIdx.x / 8)] = shared[threadIdx.x] ^ shared[threadIdx.x + 4];
	}
	__syncthreads();
}

__global__ void makep2(curandStateMRG32k3a_t *states, unsigned int *chosen, int n, float sp, unsigned int *adj){
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);
	int bit;
	unsigned int adjcache;
	unsigned int chostemp = 0;

	curandStateMRG32k3a_t state = states[thread_id];
	bit = 1;
	adjcache = adj[thread_id];

	for(int j=0; j<(sizeof(unsigned int)*8); j++){
		if(bit & adjcache){	
			chostemp += bit * (curand_uniform(&state)<sp);
		}
		bit <<= 1;
	}
	chosen[thread_id] = chostemp;

	states[thread_id] = state;
}

__global__ void makep3(curandStateMRG32k3a_t *states, int *sums2, unsigned int *rxed, unsigned int *chosen, int n, float sp, int round){
	extern __shared__ int shared1[];
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);	

	curandStateMRG32k3a_t state = states[thread_id];	//Choose your random packets
	for (int i = blockDim.x + threadIdx.x; (i< blockDim.x + (2*round)) || (i < blockDim.x + 32); i+=blockDim.x){
		shared1[i] =(int)( curand_uniform(&state) * sums2[blockIdx.x]);
	}
	__syncthreads();
	states[thread_id] = state;
	
	//Now do the replacement bit TODO: This is intentionally suboptimal, because you're never
	//choosing more than about 32 messages anyway. Really it'll be more like 5.

	int doubles = 0;
	if(threadIdx.x == 0){
		for (int test = 1; test < round; test ++){
			for (int test2 = 0; test2< test; test2 ++){
				if (shared1[blockDim.x + test] == shared1[blockDim.x + test2]){
		            		if ((doubles >= round) && (doubles >=(32 - round))) {
			                        shared1[blockDim.x + test] = (int)( curand_uniform(&state) * sums2[blockIdx.x]);
		        		}else{
    						shared1[blockDim.x + test] = shared1[blockDim.x + round + doubles];
                    			}
					doubles ++;
					test --;
					break;
				}
			}
		}
	}

	
	unsigned int acc = 0; //Now sum them up
	for (int line = (threadIdx.x % 8); line<round; line +=8){	
		acc ^= rxed[((shared1[blockDim.x + line] * n )* (n/(sizeof(unsigned int)*8)) ) + (blockIdx.x * (n/(sizeof(unsigned int)*8)) ) + (threadIdx.x / 8)]; 
	}
	shared1[threadIdx.x] = acc;
	__syncthreads();


	if(threadIdx.x % 2 ==0){
		shared1[threadIdx.x] ^= shared1[threadIdx.x + 1];
	}
	__syncthreads();
	
	if(threadIdx.x % 4 ==0){
		shared1[threadIdx.x] ^= shared1[threadIdx.x + 2];
	}
	__syncthreads();


	if(threadIdx.x % 8 ==0){	//Add to linear combinations of initial round.
		chosen[(blockIdx.x * (n/(sizeof(unsigned int)*8)) ) + (threadIdx.x / 8)] ^= shared1[threadIdx.x] ^ shared1[threadIdx.x + 4];
	}
	__syncthreads();
}

__global__ void makep4(curandStateMRG32k3a_t *states, unsigned int *chosen, int n, int *sums, float beta, unsigned int *adj){
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);
	float sp = fminf(beta* (logf(sums[blockIdx.x]) / sums[blockIdx.x]), 0.5);
	int bit;
	unsigned int adjcache;
	unsigned int chostemp = 0;

	curandStateMRG32k3a_t state = states[thread_id];
	bit = 1;
	adjcache = adj[thread_id];

	for(int j=0; j<(sizeof(unsigned int)*8); j++){
		if(bit & adjcache){	
			chostemp += bit * (curand_uniform(&state)<sp);
		}
		bit <<= 1;
	}
	chosen[thread_id] = chostemp;

	states[thread_id] = state;
}

__global__ void xmit(unsigned int *adj, unsigned int *rxed, unsigned int *chosen, int *sums, int *sums2, int n, int mix){	
	int thread_id=threadIdx.x + (blockIdx.x * blockDim.x);	
	int bit;
	int temp=sums[thread_id];
	unsigned int adjcache;

	for(int i=0; i<n/(8*sizeof(unsigned int)); i++){
		bit = 1;
		adjcache = adj[((n/(sizeof(unsigned int)*8))*thread_id)+i];
		for(int j=0; j<(sizeof(unsigned int)*8); j++){
			if(bit & adjcache){	
				for(int k=0; k<n/(8*sizeof(unsigned int)); k++){
					rxed[((temp * n) * (n/(sizeof(unsigned int)*8))) + ( (thread_id*n)/(sizeof(unsigned int)*8)) + k] 
						= chosen[(((i*8*sizeof(unsigned int)) + j)*(n/(sizeof(unsigned int)*8))) + k];
				}
				temp ++;
			}
			bit <<= 1;
		}
	}
	sums[thread_id]=temp;
	if (mix){
		sums2[thread_id]=temp;
	}
}

__global__ void diam(unsigned int *adj, unsigned int *out, int n){
	extern __shared__ unsigned int acc2[];
	unsigned int bit, acc;
	unsigned int adjcache=adj[(blockIdx.x*(n/(8*sizeof(unsigned int)))) + threadIdx.x];
	if (threadIdx.x == 0){
		out[blockIdx.x] = ~0;
	}

	for(int j = 0; j < (n/(8*sizeof(unsigned int))); j++){
		bit = 1;
		acc = 0;
		for(int i = (threadIdx.x * 8 * sizeof(unsigned int)); i<((threadIdx.x+1) * 8 * sizeof(unsigned int)); i++){
			if(adjcache & bit){
				acc |= adj[(i*(n/(8*sizeof(unsigned int)))) + j];
			}
			bit <<= 1;
		}

		acc2[threadIdx.x] = acc;
		__syncthreads();
	
		for (unsigned int i = 1; i <= (n/(16*sizeof(unsigned int))); i <<= 1){
			if ( ((threadIdx.x % (i<<1)) == 0) && (threadIdx.x + i < n/(8*sizeof(unsigned int)) ) ){
				acc2[threadIdx.x] |= acc2[threadIdx.x + i];
			}
			__syncthreads();
		}
	
		if (threadIdx.x == 0){
			out[blockIdx.x] &= acc2[0];
		}
	}
}

__global__ void linsolve(unsigned int *chosen, unsigned int *solved, int *linsums, unsigned int *adj, int *sums, int n, int setwidth){
	extern __shared__ unsigned int cacheline[];
	__shared__ int sum, set, lincache, bit, bit2, k, nei, l;
	__shared__ unsigned int adjcache;
	__shared__ int mapcache1[32];
	int linedim = n * n/(sizeof(unsigned int)*8);
	if(threadIdx.x==0){
		sum = sums[blockIdx.x];
//		set = 1;
		bit2 = 1;
		k = 0;
		l = 0;
		lincache=linsums[blockIdx.x];
		nei = 0;
		adjcache = adj[((n/(sizeof(unsigned int)*8))*blockIdx.x)];
		bit = 0;
	} 
	__syncthreads();

	int mapdim = n/(8*sizeof(unsigned int));
	int mapcache, mapcache2, lastline, temp1, temp2, linsum, enabled, nextthread, bit2c, kc, l1, setc;
	unsigned int temp3, temp4;

	mapcache1[threadIdx.x] = threadIdx.x;		//Current line you're working on (from last thread)
	mapcache2 = 32 + threadIdx.x;			//Current 'bad packet' in cache (from last cycle).
	nextthread = (int)threadIdx.x - 1;
	if (threadIdx.x==0){
		nextthread = 31;
	}
	
	while(sum>0){
		linsum = lincache;
		lincache = 0;
		enabled = 1;
		__syncthreads();
		for(int npos=(-1)*(int)threadIdx.x; npos +threadIdx.x <= linsum + 64; npos++){
			mapcache = mapcache1[nextthread];
			__syncthreads();
			mapcache1[threadIdx.x] = mapcache;

			if(npos == 0){
				if(sum > 0){
					temp2=bit;
					if (k == 0){
						set = 1;
						temp3=0;
						while (temp3 == 0){
							if (temp2 ==32){
								temp2 = 0;
								nei ++;
								adjcache = adj[((n/(sizeof(unsigned int)*8))*blockIdx.x) + nei];
							}
							if(blockIdx.x == 0){
							}
							temp3 = adjcache & (1u<<temp2);
							temp2 ++;
						}
						bit = temp2;
					}
					//TODO: cache set, bit2, l and k.
					kc = k;
					bit2c = bit2;
					setc = set;
					for(int i = 0; i < kc; i++){
						cacheline[(mapdim*mapcache2) + i] = 0;
					}
					if(l != 0){
						temp4=0;
						l1=l;
						while(l1 < 8*sizeof(unsigned int)){
							l1++;
							temp4 += chosen[(((temp2-1)+(nei*(8*sizeof(unsigned int)))) * mapdim) + kc] & bit2c;
							bit2c <<= 1;
						}
						cacheline[(mapdim*mapcache2) + kc] = temp2;
						kc++;
					}
					while(kc<n/(8*sizeof(unsigned int)) and ((kc*8*sizeof(unsigned int)) < (setc*setwidth)) ){ 
						cacheline[(mapdim*mapcache2) + kc] = chosen[(((temp2-1)+(nei*(8*sizeof(unsigned int)))) * mapdim) + kc];
						kc++;
					}
					if (((kc<(n/(8*sizeof(int)))) && (kc*8*sizeof(unsigned int)) != (setc*setwidth))){
						bit2c = 1;
						temp4=0;
						for(l1 = 0; l1 < (((kc+1)*8*sizeof(unsigned int))-(setc*setwidth)); l1++){
							temp4 += chosen[(((temp2-1)+(nei*(8*sizeof(unsigned int)))) * mapdim) + kc] & bit2c;
							bit2c <<= 1;
						}
						l = l1;
						cacheline[(mapdim*mapcache2) + kc] = temp4;
					} else if(kc< (n/(8*sizeof(unsigned int)))){
						cacheline[(mapdim*mapcache2) + kc] = 0;
					}
					set = setc + 1;
					for(int i = kc+1; i<(n/(8*sizeof(unsigned int))); i++){
						cacheline[(mapdim*mapcache2) + i] = 0;
					}
					if(kc>= (n/(8*sizeof(unsigned int)))){
						sum --;
						l = 0;
						bit2 = 1;
						k = 0;
					}else{
						k = kc;
						bit2 = bit2c;
					}
				
				
					lastline = ((sum == 0) || (threadIdx.x == 31));
				}else{
					enabled = 0;
				}
			}

			if((npos >= 0) && enabled){ 
				if(threadIdx.x == 0){
					if (npos >= linsum){
						for(int i=0; i<mapdim; i++){
							cacheline[(mapdim*mapcache) + i] = 0;
						}
					}else{
						for(int i=0; i<mapdim; i++){
							cacheline[(mapdim*mapcache) + i] = solved[(npos * linedim) + (blockIdx.x * mapdim) + i];
						}
					}
				}
				temp1 = 0;
				while((cacheline[(mapdim*mapcache) + temp1]==0) &&( cacheline[(mapdim*mapcache2) + temp1] ==0) && ((1+temp1)<mapdim)){
					temp1++;
				}

				if(cacheline[(mapdim*mapcache) + temp1] < cacheline[(mapdim*mapcache2) + temp1]){
					temp2 = mapcache2;
					mapcache1[threadIdx.x] = mapcache2;
					mapcache2 = mapcache;
					mapcache = temp2;
				}

				if((cacheline[(mapdim*mapcache) + temp1] ^ cacheline[(mapdim*mapcache2) + temp1]) < cacheline[(mapdim*mapcache2) + temp1]){
					while(temp1 < mapdim){
						cacheline[(mapdim*mapcache2) + temp1] ^= cacheline[(mapdim*mapcache) + temp1]; 
						temp1 ++;
					}
				}

				if(lastline){
					if ((lincache<n) && (npos < n) ){
						temp2=0;
						for(int i=0; i<mapdim; i++){
							temp2 = temp2 || (cacheline[(mapdim*mapcache) + i]>0);
						}
						if (temp2){
							for(int i=0; i<mapdim; i++){
								solved[(lincache * linedim) + (blockIdx.x * mapdim) + i] = cacheline[(mapdim*mapcache) + i];
							}
						}
						lincache += temp2;
					}
				}
	
		
			}
			__syncthreads();
		}
	}
	if (threadIdx.x==0){
		linsums[blockIdx.x] = lincache;
	}
}

void makepacket(int mix, int rounds, float p, int n, curandStateMRG32k3a_t *packstates, unsigned int *chosen, unsigned int *adj, int *sums2, unsigned int *rxed, unsigned int *base_rxed0, unsigned int *base_rxed1, int *linsums, float betap){
	//
	float sp, sp2;

	switch(mix){
		case 5:		//Code over initial buffer only, using pi=log(#in-neighbours) / #in-neighbours. Chapter 7 thesis/Transactions IT algorithm B.
			makep4<<<n, n/(8*sizeof(unsigned int)), (3 * n * sizeof(int)) +((n*sizeof(int))/sizeof(unsigned int))>>>(packstates, chosen, n, sums2, betap, adj);
			break;
		case 4:		//Chapter 6 thesis/Allerton algorithm.
				//Graph evolution NOT ALLOWED for this algorithm (due to
				//implementation)
			sp2 = min(0.5f, (  (ceil((1.0f-p)/p) *((log(n)/log(2)) + 1.0f)) / (n*p*p)));
			makep2<<<n, n/(8*sizeof(unsigned int)), (3 * n * sizeof(int)) +((n*sizeof(int))/sizeof(unsigned int))>>>(packstates, chosen, n, sp2, adj);
			break;

		case 3:		//Code over initial buffer for 1/p rounds, and then over everything.
			sp = min((betap*log(n))/(n*p), 0.5);
			if(rounds < (1.0f/p)){
				makep2<<<n, n/(8*sizeof(unsigned int)), (3 * n * sizeof(int)) +((n*sizeof(int))/sizeof(unsigned int))>>>(packstates, chosen, n, sp, adj);
				xmit<<<n/32,32>>>(adj, rxed, chosen, sums2, sums2, n, 0);
			} else{
				makep<<<n, n/sizeof(unsigned int), (3 * n * sizeof(unsigned int)) +((n*sizeof(int))/sizeof(unsigned int)) >>>(packstates, sums2, rxed, chosen, n, sp);
				xmit<<<n/32,32>>>(adj, rxed, chosen, sums2, sums2, n, 0);
			}
			break;

		case 2:		//Code over initial buffer, plus i mixed packets in round i+1.
			sp = min((betap*log(n))/(n*p), 0.5);
			makep2<<<n, n/(8*sizeof(unsigned int)), (3 * n * sizeof(int)) +((n*sizeof(int))/sizeof(unsigned int))>>>(packstates, chosen, n, sp, adj);
			if (rounds>1){
				makep3<<<n, n/sizeof(unsigned int), (max(32,2*rounds)* sizeof(unsigned int)) +((n*sizeof(int))/sizeof(unsigned int)) >>>(packstates, sums2, rxed, chosen, n, sp, rounds);
			}
			xmit<<<n/32,32>>>(adj, rxed, chosen, sums2, sums2, n, 0);
			break;
		case 1:		//Code over everything you ever hear.
			sp = min((betap*log(n))/(n*p), 0.5);
			makep<<<n, n/sizeof(unsigned int), (3 * n * sizeof(unsigned int)) +((n*sizeof(int))/sizeof(unsigned int)) >>>(packstates, sums2, rxed, chosen, n, sp);
			xmit<<<n/32,32>>>(adj, rxed, chosen, sums2, sums2, n, 0);
			break;
		case 0:		//Code over initial buffer only.
			sp = min((betap*log(n))/(n*p), 0.5);
			makep2<<<n, n/(8*sizeof(unsigned int)), (3 * n * sizeof(int)) +((n*sizeof(int))/sizeof(unsigned int))>>>(packstates, chosen, n, sp, adj);
			break;
		case -1:	//Randomly forward your entire buffer, which you're building up over time. (Chapter 6 Thesis Algorithm/Algorithm A2 Transactions IT)
			if(n >= 1024){
				choosep<<<n/1024,32>>>(packstates, linsums, base_rxed1, chosen,  n, p);
			}else{
				small_choosep<<<n/32,32>>>(packstates, linsums, base_rxed1, chosen,  n, p);
			}
			break;
		case -2:	//Randomly forward messages from the buffer of packets received in the first round only. (Algorithm A1 Transactions IT)
			if(n >= 1024){
				choosep<<<n/1024,32>>>(packstates, sums2, base_rxed0, chosen,  n, p);
			}else{
				small_choosep<<<n/32,32>>>(packstates, sums2, base_rxed0, chosen,  n, p);
			}
			break;
	}
	return;
}

void send_p1(int mix, float p, float betap, int n, curandStateMRG32k3a_t *packstates, unsigned int *chosen, unsigned int *adj, unsigned int *adj2, int *sums, int *sums2, unsigned int *rxed, unsigned int* solved, float alpha, unsigned int *base_rxed0, unsigned int *base_rxed1, int *linsums){
	if(mix < 0){
		//Send 1st packet for baseline and check missing ones (gpu side)
		if(n>=1024){
			rxinit<<<n/1024,32>>>(base_rxed0, base_rxed1, linsums, n, p);
			choosep<<<n/1024,32>>>(packstates, linsums, base_rxed0, chosen,  n, p);
			baseline_xmit<<<n/1024,32>>>(adj, base_rxed0, chosen,  n, p);
			baseline_xmit<<<n/1024,32>>>(adj, base_rxed1, chosen,  n, p); //TODO: May improve efficiency by just copying.
			upsums<<<n/1024,32>>>(base_rxed1, linsums, sums2, n);
		}else{
			small_init<<<n/32,32>>>(base_rxed0, base_rxed1, linsums, n, p);
			small_choosep<<<n/32,32>>>(packstates, linsums, base_rxed0, chosen,  n, p);
			small_xmit<<<n/32,32>>>(adj, base_rxed0, chosen,  n, p);
			small_xmit<<<n/32,32>>>(adj, base_rxed1, chosen,  n, p); //TODO: May improve efficiency by just copying.
			small_upsums<<<n/32,32>>>(base_rxed1, linsums, sums2, n);
		}
	} else {//Send the first network coded packet and run linear solver
		init<<<n/32,32>>>(adj, adj2, sums, sums2, rxed, chosen, solved, linsums, n, (alpha > 0)); //packet 1
				
		if((mix ==2) || (mix == 0)){
			newsums<<<n/32, 32>>>(adj, sums, sums2, n);
		}else{
			xmit<<<n/32,32>>>(adj, rxed, chosen, sums, sums2, n, mix);		//Build buffer		TODO: Increase the number of threads to improve occupancy.
		}

		linsolve<<<n,32,(64*sizeof(int)*n)/(8*sizeof(unsigned int))>>>(chosen, solved, linsums, adj, sums, n, n);
	}

	return;
}

int main(int argc, char *argv[])
{
	if (argc!=9){
		printf("Usage: alpha sims n mix beta seed gpu path.\n");
		return 1;
	}

	omp_set_num_threads(4);
	float alpha = atof(argv[1]);
	int mix = atoi(argv[4]);

	if((alpha > 0) && (mix == 4)){
		printf("Error: Markov chain graph evolution not allowed for the Allerton method.\n");
		return 1;
	}

	if (mix > 5){
		printf("Error: invalid mix, mix must lie in -2 <= mix <= 5\n");
		return 1;
	}

	float p[6] = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4};	//These MUST be in decreasing order, as this does graph thinning.
	int sims=atoi(argv[2]);
	float betap = atof(argv[5]);

	int n = atoi(argv[3]);
	int seed = atoi(argv[6]);
	check(cudaSetDevice(atoi(argv[7])));

	int results[sims][sizeof(p)/sizeof(p[0])][2] = {0}; //each element is [#rounds, iswide?]
	
	float beta; //This is for the Markov chain evolution, not the density constant.
	int success;
	float sp;
	
	time_t start= (time(NULL));

	int *sums, *sums2, *linsums, *hlinsums, *new1, rounds, ns;
	unsigned int *adj, *adj2, *rxed, *solved, *chosen, *out, *base_rxed[2];
	//TODO: Be aware that variables like chosen may be used for different things. (Like an array
	//of n ints vs an nxn array of bits for coding coeffs)
	curandStateMRG32k3a_t *graphstates, *packstates;

	//Initialise all the device variables

	cudaMalloc((void**) &adj, ((long)n*(long)n*0.125));
	if(alpha > 0){
		cudaMalloc((void**) &adj2, ((long)n*(long)n*0.125));
	} else {
		adj2=adj;
	}

	cudaMalloc((void**) &linsums, n*sizeof(int));
	hlinsums = (int*) malloc(n * sizeof(int));
	cudaMalloc((void**) &graphstates, (n*n*sizeof(curandStateMRG32k3a_t))/(8*sizeof(unsigned int)));
	device_init<<<(n*n)/(32*8*sizeof(unsigned int)),32>>>(graphstates, seed);//Initialisation for the graph

       	if (mix < 0){
		//Init for the baseline
		int scale = 1;
		if(n>=1024){
			scale = 32;
		}
		cudaMalloc((void**) &packstates, (n/scale)*sizeof(curandStateMRG32k3a_t));
		device_init<<<n/(32*scale),32>>>(packstates, seed + 1000);

		cudaMalloc((void**) &base_rxed[0], (n*n*sizeof(int))/scale);
		cudaMalloc((void**) &base_rxed[1], (n*n*sizeof(int))/scale);
		cudaMalloc((void**) &chosen, n*sizeof(int));
		if(mix == -2){
			cudaMalloc((void**) &sums2, n*sizeof(int));
		} else {
			sums2=linsums;
		}
	} else {
		//Init for network coding
		cudaMalloc((void**) &packstates, ((n*n)/sizeof(unsigned int))*sizeof(curandStateMRG32k3a_t)); 
		cudaMalloc((void**) &rxed, 20* ((long)n*(long)n*(long)n*0.125)); 
		cudaMalloc((void**) &solved,( ((long)n*(long)n*(long)n)/8));
		cudaMalloc((void**) &sums, n*sizeof(int));
		if ((mix != 0) && (mix != 4)){
			cudaMalloc((void**) &sums2, n*sizeof(int));
		} else {
			sums2=sums;
		}
		cudaMalloc((void**) &new1, n*sizeof(int));
		cudaMalloc((void**) &chosen, (n*n)/8);


		device_init<<<(n*n)/(32*sizeof(unsigned int)), 32>>>(packstates, seed + 1000);
	}	


	for (int sim=0; sim<sims; sim++){
		for (int pi=0;	pi < sizeof(p)/sizeof(p[0]); pi++){
			beta = ((1-p[pi])/p[pi])*alpha; //This is for the Markov chain evolution, not the density constant.
			if(mix == 4){
				ns= (int)ceil(n/ ceil((1.0f-p[pi])/p[pi]));
			}
			if(pi==0){
				graphgen<<<(n*n)/(32*8*sizeof(unsigned int)),32>>>(graphstates, adj, adj2, n, p[pi], (alpha > 0)); //Initial graph
			}else{
				graphthin<<<(n*n)/(32*8*sizeof(unsigned int)),32>>>(graphstates, adj, adj2, n, p[pi]/p[pi-1], (alpha > 0)); //Thin the graph
			}

			send_p1(mix, p[pi], betap, n, packstates, chosen, adj, adj2, sums, sums2, rxed, solved, alpha, base_rxed[0], base_rxed[1], linsums);	//Send first packet
			rounds = 1;
			
			printf("\rn=%d, p=%f simulation %d, i=%d", n, p[pi], sim, rounds);
			fflush(stdout);

			if(mix==4){
				check(cudaMemcpy(hlinsums, linsums, sizeof(int)*n, cudaMemcpyDeviceToHost));
																						
				sp = min(1.0f, (log(ns-ceil((1.0f-p[pi])/p[pi]))/(p[pi]*(ns-ceil((1.0f-p[pi])/p[pi])))));
				makep2<<<n, n/(8*sizeof(unsigned int)), (3 * n * sizeof(unsigned int)) +((n*sizeof(int))/sizeof(unsigned int))>>>(packstates, chosen, n, sp, adj); //Partitioned Allerton rounds
				success = 1;
#pragma omp parallel for reduction (&:success)
				for(int i=0; i<n; i++){
					success &= (hlinsums[i] == n);
				}
	
				if(!success){
					rounds =(int) ceil(1.0f/p[pi]);
					printf("\rn=%d, p=%f simulation %d, i=%d", n, p[pi], sim, rounds);
					fflush(stdout);
					linsolve<<<n,32,(64*sizeof(int)*n)/(8*sizeof(unsigned int))>>>(chosen, solved, linsums, adj, sums, n, ns);
				}
			}else{
				success = 0;
			}
			
			while(!success && ((alpha>0) || ((mix > 0) && (mix < 4)) ||((mix == -1) && (rounds < 5000)) || (rounds < (1.0f/p[pi]) + 4) || ((mix == -2) && (rounds < 300)) ) ){
				check(cudaMemcpy(hlinsums, linsums, sizeof(int)*n, cudaMemcpyDeviceToHost));
	
				makepacket(mix, rounds, p[pi], n, packstates, chosen, adj, sums2, rxed, base_rxed[0], base_rxed[1], linsums, betap); //Make a new packet whilst you check to see if you're done, in case you need it.
	
				success = 1;
#pragma omp parallel for reduction (&:success)
				for(int i=0; i<n; i++){
					success &= (hlinsums[i] == n);
				}
				if (!success){
					rounds ++;
					printf("\rn=%d, p=%f simulation %d, i=%d", n, p[pi], sim, rounds);
					fflush(stdout);
					if(alpha>0){
						graphevo<<<(n*n)/(32*8*sizeof(unsigned int)),32>>>(graphstates, adj2, n, alpha, beta);
						if(mix >= 0){
							newsums<<<n/32, 32>>>(adj2, sums, sums, n); //Update sums for network coding after graph evolution
						}
					}
					if(mix >= 0){
						//Do the linear solve if you're network coding
						linsolve<<<n,32,(64*sizeof(int)*n)/(8*sizeof(unsigned int))>>>(chosen, solved, linsums, adj2, sums, n, n);
					}else{
						//Transmit the packet and check how many each node
						//now have missing otherwise.
						if(n >= 1024){
							baseline_xmit<<<n/1024,32>>>(adj2, base_rxed[1], chosen,  n, p[pi]);
							upsums<<<n/1024,32>>>(base_rxed[1], linsums, linsums, n);
						}else{
							small_xmit<<<n/32,32>>>(adj2, base_rxed[1], chosen,  n, p[pi]);
							small_upsums<<<n/32,32>>>(base_rxed[1], linsums, linsums, n);
						}
					}
				}
			}
			if(!success){
				cudaMalloc((void**) &out, n*sizeof(unsigned int));
				//If you get down here, you exceeded the max # runs, so check the
				//graph diameter
				diam<<<n, n/(8*sizeof(unsigned int)), n/8>>>(adj, out, n);
				cudaError_t thetest = cudaDeviceSynchronize();
				check(cudaMemcpy(hlinsums, out, sizeof(int)*n, cudaMemcpyDeviceToHost));
				success = 1;
#pragma omp parallel for reduction (&:success)
				for(int i=0; i<n; i++){
					success &= (!(~hlinsums[i]));
				}
				if(success){//Then just carry on.
					success = 0;
					while(!success){
						check(cudaMemcpy(hlinsums, linsums, sizeof(int)*n, cudaMemcpyDeviceToHost));
			
						makepacket(mix, rounds, p[pi], n, packstates, chosen, adj, sums2, rxed, base_rxed[0], base_rxed[1], linsums, betap); //Make a new packet whilst you check to see if you're done, in case you need it.
			
						success = 1;
#pragma omp parallel for reduction (&:success)
						for(int i=0; i<n; i++){
							success &= (hlinsums[i] == n);
						}
						if (!success){
							rounds ++;
							printf("\rn=%d, p=%f simulation %d, i=%d", n, p[pi], sim, rounds);
							fflush(stdout);
							if(alpha>0){
								graphevo<<<(n*n)/(32*8*sizeof(unsigned int)),32>>>(graphstates, adj2, n, alpha, beta);
								if(mix >= 0){
									newsums<<<n/32, 32>>>(adj2, sums, sums, n); //Update sums for network coding after graph evolution
								}
							}
							if(mix >= 0){
								//Do the linear solve if you're network coding
								linsolve<<<n,32,(64*sizeof(int)*n)/(8*sizeof(unsigned int))>>>(chosen, solved, linsums, adj2, sums, n, n);
							}else{
								//Transmit the packet and check how many each node
								//now have missing otherwise.
								if(n >= 1024){
									baseline_xmit<<<n/1024,32>>>(adj2, base_rxed[1], chosen,  n, p[pi]);
									upsums<<<n/1024,32>>>(base_rxed[1], linsums, linsums, n);
								}else{
									small_xmit<<<n/32,32>>>(adj2, base_rxed[1], chosen,  n, p[pi]);
									small_upsums<<<n/32,32>>>(base_rxed[1], linsums, linsums, n);
								}
							}
						}
					}
				}
			}
			results[sim][pi][0] = rounds;
			results[sim][pi][1] = !success;
		}
	}

	cudaFree(adj);
	if(alpha > 0){
		cudaFree(adj2);
	}
	cudaFree(graphstates);
	cudaFree(packstates);
	cudaFree(base_rxed[0]);
	cudaFree(base_rxed[1]);
	cudaFree(rxed);
	cudaFree(solved);
	cudaFree(sums);
	cudaFree(linsums);
	cudaFree(chosen);
	cudaFree(new1);
	cudaFree(out);
	if((mix > 0) && (mix != 4)){
		cudaFree(sums2);
	}
	
	free(hlinsums);
	
	float duration = ((time(NULL)) - start);
	int hours = duration/3600;
	int minutes = (duration-(3600*hours))/60;
	float seconds = duration - ((3600*hours) + (60*minutes));
	printf("\r____________Run time: %d hours, %d minutes and %f seconds.____________\n", hours, minutes, seconds);
	
	char path[PATH_MAX + 50];
	getcwd(path, sizeof(path));
	strcat(path, "/");
	strcat(path, argv[8]);

	FILE* fp;
	fp=fopen(path, "w");
	fprintf(fp, "Timesteps,diam>2?,p,alpha,beta,n,seed,mix\n");
	for(int pi=0; pi<sizeof(p)/sizeof(p[0]); pi++){
		for(int sim=0; sim<sims; sim++){
			fprintf(fp, "%d,%d,%f,%f,%f,%d,%d,%d\n",results[sim][pi][0], results[sim][pi][1], p[pi], alpha, betap, n, seed, mix);
		}
	}
	fclose(fp);

	return(0);
}
