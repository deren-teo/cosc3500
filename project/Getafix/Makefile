CXX=g++
CXXFLAGS=-std=c++11 -Wall -pedantic -mavx -fopenmp
NVCC=nvcc
NVFLAGS=--gpu-architecture=sm_35 -Wno-deprecated-gpu-targets

TARGETS = game_of_life

default : all

$(TARGETS) : gridEvolve.h gridEvolve.o gridEvolveGPU.cuh gridEvolveGPU.o parser.h parser.o

LIBS_game_of_life = -lgomp -lm -ldl -lcudart

CXXFLAGS_game_of_life =

gridEvolve.o : gridEvolve.h
gridEvolveGPU.o : gridEvolveGPU.cuh
parser.o : parser.h

%GPU.o : %GPU.cu
	$(NVCC) $(OFLAG) $(NVFLAGS) $(NVFLAGS_$(basename $<)) -c $< -o $@

% : %.cu
	$(NVCC) $(OFLAG) $(NVFLAGS) $(NVFLAGS_$@) $(filter %.o %.cu, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

%.o : %.cpp
	$(CXX) $(OFLAG) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@

% : %.cpp
	$(CXX) $(OFLAG) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

all : $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean default all
