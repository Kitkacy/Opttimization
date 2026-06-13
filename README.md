Max-cut is implemented in C++: full bruteforce; greedy algorithm and greedy clustering for genetic algorithm - not fully implemented, yet gives right results for sparse graphs. Goemans-Williamson algorithm is working incorrectly in C++ and is working in Python. However, Python code is taken from the video you have shared and is modified to parse the input format.
```
Output example:
     0     0     0    40     0
     0     0     0     0     0
     0     0     0    30    30
    40     0    30     0     0
     0     0    30     0     0

Results:
         Solver Name   |    Time (ms) |      Max-Cut | Solution
-----------------------|--------------|--------------|------------------
          BruteForce   |        0.003 |          100 | 1 0 1 0 0 
  Goemans-Williamson   |        0.026 |            0 | 1 1 1 1 1 
      Greedy-by-node   |        0.005 |          100 | 0 0 0 1 1 
Greedy-by-node+genetic |        0.004 |          100 | 0 -1 0 1 1 
```
