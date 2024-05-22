CC         = g++
CFLAGS     = -c -O3 -fopenmp
LDFLAGS    = -fopenmp

MAIN_SRC   = main.cpp
VIS_SRC    = visibility.cpp

MAIN_OBJ   = $(MAIN_SRC:.cpp=.o)
VIS_OBJ    = $(VIS_SRC:.cpp=.o)

all: main visibility

main: $(MAIN_OBJ)
	$(CC) $(MAIN_OBJ) $(LDFLAGS) -o $@

visibility: $(VIS_OBJ)
	$(CC) $(VIS_OBJ) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(MAIN_OBJ) $(VIS_OBJ) main visibility

