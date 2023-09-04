CFLAGS = -std=c++11 -Wall -pedantic

.PHONY = all game_of_life clean

all: game_of_life

game_of_life: Makefile game_of_life.cc parser.h parser.cc
	g++ $(CFLAGS) -O3 game_of_life.cc parser.cc -o game_of_life

debug: Makefile game_of_life.cc parser.h parser.cc
	g++ $(CFLAGS) -O3 -g game_of_life.cc parser.cc -o game_of_life

prof: Makefile game_of_life.cc parser.h parser.cc
	g++ $(CFLAGS) -O0 -pg game_of_life.cc parser.h parser.cc -o game_of_life

clean:
	rm -rf *.o
	rm -rf game_of_life
