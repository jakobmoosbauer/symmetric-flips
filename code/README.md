#How to run MatrixMult22.py

The program runs the symmetric flip graph search algorithm to find fast matrix multiplication schemes.

It is written in Python with a C++ solver.  You need to compile the standalone C++ program FlipSolver22.cpp (with full optimisation on), and provide the Python script with the location of the executable.  To do that, in MatrixMult22.py, modify the line:

fastsolver='C:/Flip Graphs/FlipSolver22/x64/Release/FlipSolver22.exe'

Without the C++ solver, the Python program will still run, but it will be much, much slower.

The program as default will run a test case for the 4x4 case (attempting to find a solution with rank 49, it won't every single time), to try this, type:

python3 MatrixMult22.py

The program normally takes an input file, to run the 5x5 case, type:

python3 MatrixMult22.py r5-93-1.txt

The file r5-93-1.txt details the case to be run, this one will try 100 runs to find the rank 93 solution, it will find it (at random) about 1 in 30 attempts. Depending on hardware, with the C++ solver, each run of this case might take about 10 seconds.  

The file r6-153-1.txt is an example input file for the 6x6 case. This case finds
the target rank in about 1 in 1500 attempts.

We also provide to input files with seeds that allow to find the target rank after 11(5x5) and 17(6x6) runs.
