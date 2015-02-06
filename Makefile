CC=g++
CFLAGS=-c -Wall
LDFLAGS=
SOURCES=src/neuron.cpp src/neural_network.cpp src/training_data.cpp src/main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=neural_network

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean: 
	rm -rf src/*o $(EXECUTABLE)
