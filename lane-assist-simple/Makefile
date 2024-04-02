
CXX = g++
CC = gcc
NVCC = nvcc

#OpenCV
CXX_OPENCV_FLAGS+=`pkg-config opencv --cflags`
LD_OPENCV_FLAGS+=`pkg-config opencv --libs`


CFLAGS=-O0 -I. -g
CXXFLAGS=-O0 -I.
NVFLAGS=$(CFLAGS) -G

LIBS =-lpng -lm -lcuda -lcudart

SRC = png_io.o routinesCPU.o routinesGPU.o main.o
	
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $<


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

image: $(SRC) 
	$(CXX) -o image  $(SRC) $(CXXFLAGS) $(LIBS) 

clean:
	rm -f *.o image
