#! /usr/bin/env python3

import sqlite3
import numpy as np
from matplotlib import pyplot as plt
import argparse
import math
import copy

def forward1(x):
    return 1/x

def inverse1(x):
    return 1/x

def width1(x):
    width = 0.1
    return ((x/(1+x*width)) - x)

def forward2(x):
    return 1/(x**2)

def inverse2(x):
    return 1/np.sqrt(x)

def width2(x):
    width = 0.5
    return np.sqrt(((x**2)/(1 + ((x**2)*width))))-x

parser=argparse.ArgumentParser(description="Generate Monte Carlo simulation graphs for Chapter 7 of the PhD Thesis, comparing the RLNC algorithm with the baseline.")

parser.add_argument("dbpath", type=str, help="Path to sqlite database")
args=parser.parse_args()

conn = sqlite3.connect(args.dbpath)
cursor = conn.cursor()

p = ["0.900000", "0.800000", "0.700000", "0.600000", "0.500000", "0.400000"]
pfloat = np.asarray(p, dtype=float)

################################################################################
#####################################GRAPHS#####################################
################################################################################
colours = ["green", "blue", "orange"]
if input("Run: timesteps vs n? y/return: ") == "y":
    for pi in p:
        for betai in range(3):
            beta=2**betai
            data = []
            nvalues = []
            for ni in range(6, 11):
                n = 2**ni
                cursor.execute("""SELECT Timesteps FROM final WHERE n=""" + str(n) + """ and p='""" + pi + """' and alpha="0.000000" and mix=5 and beta='""" + str(beta) + """.000000'""")
                data.append(list(map(int,*zip(*cursor.fetchall()))))# , label="n = " + str(n))
                nvalues.append(n)
            print("p = " + str(pi) + ", beta = " + str(beta))
            fig=plt.figure()
            ax = fig.add_subplot(111)
            bp1=ax.boxplot(data, showfliers=False, showcaps=False, boxprops=dict(color="blue"), whiskerprops=dict(color="blue"), medianprops=dict(color="blue"))
            #ax.plot(range(1, 6), ((1/float(pi))+2)*np.ones(5) , color=colours[betai])
            plt.ylabel("Transmission time-steps")
            plt.xlabel("n")
            mix = 1
            data = []
            nvalues = []
            for ni in range(6, 11):
                n = 2**ni
                cursor.execute("""SELECT Timesteps FROM final WHERE n=""" + str(n) + """ and p='""" + pi + """' and alpha="0.000000" and mix=""" + str((-1*mix)))
                data.append(list(map(int,*zip(*cursor.fetchall()))))# , label="n = " + str(n))
                nvalues.append(n)
            print("p = " + str(pi) + ", mix = -1")
            bp2=ax.boxplot(data, showfliers=False, showcaps=False, boxprops=dict(color="green"), whiskerprops=dict(color="green"), medianprops=dict(color="green"))
            ax.set_xticklabels(nvalues)
            plt.legend([bp2["boxes"][0], bp1["boxes"][0]], ["Forwarding", "Coding"])
            plt.show()

conn.close()
