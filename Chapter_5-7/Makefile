APPS=hello

all: ${APPS}

%: %.cu
        nvcc -std=c++14 -Xcompiler -fopenmp -lcuda -lcudart -lcurand -lgomp -rdc=true -lcudadevrt -o $@ $<
clean:
        rm -f ${APPS}

