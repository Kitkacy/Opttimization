CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
TARGET   = ga_main
SRC      = main.cpp
INSTANCE = t2g10_5555.txt
CONFIG   = ga_config.txt

.PHONY: all run test clean rebuild

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET) $(INSTANCE) $(CONFIG)

test: $(TARGET)
	python3 run_tests.py
	python3 generate_figures.py

pdf:
	pdflatex Documentation.tex
	pdflatex Documentation.tex

clean:
	rm -f $(TARGET)

rebuild: clean all
