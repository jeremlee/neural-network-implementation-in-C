CC = clang
CFLAGS = -Wall -Wextra -std=c99 -Iinc

TARGET = bin/neural_network
SRC = src/main.c src/neuralnetwork.c
OBJ = $(SRC:src/%.c=bin/%.o)
LINK = -lm

all: bin $(TARGET) run

bin:
	mkdir -p bin

bin/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@ $(LINK)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf bin
