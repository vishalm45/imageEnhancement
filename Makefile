# Makefile for OpenCL Histogram Equalisation project

# Compiler and flags
CXX = g++
CXXFLAGS = -O2 -Wall -std=c++11

# Libraries
LDLIBS = -lOpenCL -lX11 -lpthread

# Files
SRC = submission.cpp
OUT = submission

# Default rule
all: $(OUT)

# Compile
$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(OUT) $(SRC) $(LDLIBS)


