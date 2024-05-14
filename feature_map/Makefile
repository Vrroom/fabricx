# calls:
CC         = g++
CFLAGS     = -c -O3 -fopenmp
LDFLAGS    = -fopenmp
EXECUTABLE = main

SOURCES    = main.cpp
OBJECTS    = $(SOURCES:.cpp=.o)

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o main

