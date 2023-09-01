CFLAGS = -Wall -pedantic -o3

.PHONY = all game_of_life clean

all: game_of_life

game_of_life: Makefile game_of_life.cc parser.h parser.cc
	g++ $(CFLAGS) game_of_life.cc parser.cc -o game_of_life

debug: Makefile game_of_life.cc parser.h parser.cc
	g++ $(CFLAGS) -g game_of_life.cc parser.cc -o game_of_life

clean:
	rm -rf *.o
	rm -rf game_of_life
