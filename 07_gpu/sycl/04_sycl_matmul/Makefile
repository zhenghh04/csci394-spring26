CXX ?= icpx
CXXFLAGS ?= -O2 -std=c++17 -fsycl

all: app

app: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f app
