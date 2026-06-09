CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
TARGET   = ga_main
SRC      = main.cpp
INSTANCE = t2g10_5555.txt
CONFIG   = ga_config.txt

.PHONY: all run clean rebuild

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET) $(INSTANCE) $(CONFIG)

clean:
	rm -f $(TARGET)

rebuild: clean all
