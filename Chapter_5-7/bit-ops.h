/*
You got this code from:

http://www.mathcs.emory.edu/~cheung/Courses/255/Syllabus/1-C-intro/bit-array.html

and then you added PutBit
*/

//For arrays:
#define PutaBit(A,k,setting)     ( A[(k/32)] =(A[(k/32)] & ~(1 << (k%32))) + (setting << (k%32)) )
#define SetaBit(A,k)     ( A[(k/32)] |= (1 << (k%32)) )
#define ClearaBit(A,k)   ( A[(k/32)] &= ~(1 << (k%32)) )
#define TestaBit(A,k)    ( A[(k/32)] & (1 << (k%32)) )

//For ints (I added this)
#define PutBit(A,k,setting)     ( A =(A & ~(1 << k)) + (setting << k) )
#define SetBit(A,k)     ( A |= (1 << k) )
#define ClearBit(A,k)   ( A &= ~(1 << k) )
#define TestBit(A,k)    (( A & (1 << k) )!=0)
