# Chapter 5-7 (allcast) code

Code used to produce Monte Carlo simulations for the allcast algorithms in my PhD thesis.

Final.cu is written in CUDA C, and once compiled may be used to generate Monte Carlo simulations.
This program requires an nVIDIA card with a *LOT* of GPU RAM.

The python scripts may be used to produce the graphics in the thesis from simulation data. The data
must be imported to an appropriately named table in an SQLITE database from the code's CSV output.
The code may contain graphs which did not appear in the thesis.
