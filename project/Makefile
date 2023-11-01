CXX=g++
CXXFLAGS=-std=c++11 -Wall -pedantic -lgomp -lm -ldl -mavx -fopenmp

TARGETS = game_of_life

default : all

game_of_life: Makefile game_of_life.cc gridEvolve.h gridEvolve.cpp parser.h parser.cc
	$(CXX) $(CXXFLAGS) $(OFLAG) game_of_life.cc gridEvolve.cpp parser.cc -o game_of_life

prof: Makefile game_of_life.cc parser.h parser.cc
	$(CXX) $(CXXFLAGS) -O0 -pg game_of_life.cc parser.h parser.cc -o game_of_life

all : $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean default all
