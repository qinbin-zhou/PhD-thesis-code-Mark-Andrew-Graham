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

parser=argparse.ArgumentParser(description="Generate Monte Carlo simulation graphs for Chapter 6 of the PhD Thesis.")

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
        betai=0
        beta=2**betai
        data = []
        nvalues = []
        for ni in range(6, 11):
            n = 2**ni
            cursor.execute("""SELECT Timesteps FROM final WHERE n=""" + str(n) + """ and p='""" + pi + """' and alpha="0.000000" and mix=4 and beta='""" + str(beta) + """.000000'""")
            data.append(list(map(int,*zip(*cursor.fetchall()))))# , label="n = " + str(n))
            nvalues.append(n)
        print("p = " + str(pi) + ", beta = " + str(beta))
        fig=plt.figure()
        ax = fig.add_subplot(111)
        bp=ax.boxplot(data, showfliers=False, showcaps=False, boxprops=dict(color=colours[betai]), whiskerprops=dict(color=colours[betai]), medianprops=dict(color=colours[betai]))
        lin=ax.plot(range(1, 6), (math.ceil(1/float(pi))+1)*np.ones(5) , color=colours[2], linestyle=(0, (5, 10)), label="Theory")
        ax.set_xticklabels(nvalues)
        plt.ylabel("Transmission time-steps")
        plt.xlabel("n")
        plt.legend([bp["boxes"][0], lin[0]], ["Simulations", "Theory"])
        plt.show()


if input("Run: timesteps vs p? y/return: ") == "y":
    for ni in range(6, 11):
        betai=0
        beta = 2**betai
        data = []
        for pi in p:
            n = 2**ni
            cursor.execute("""SELECT Timesteps FROM final WHERE n=""" + str(n) + """ and p='""" + pi + """' and alpha="0.000000" and mix=4 and beta='""" + str(beta) + """.000000'""")
            data.append(list(map(int,*zip(*cursor.fetchall()))))# , label="n = " + str(n))
        print("n = " + str(n) + ", beta = " + str(beta))
        fig=plt.figure()
        ax = fig.add_subplot(111)
        bp=ax.boxplot(data, showfliers=False, showcaps=False, boxprops=dict(color=colours[betai]), whiskerprops=dict(color=colours[betai]), medianprops=dict(color=colours[betai]), positions=pfloat, widths=width1(pfloat))
        lin=ax.step(pfloat, np.ceil(1/pfloat) + 1, color=colours[2], linestyle =(0, (5, 10)))
        ax.set_xscale('function', functions=(forward1, inverse1))
        plt.xlim(0.38,1)
        plt.ylabel("Transmission time-steps")
        plt.xlabel("p")
        plt.legend([bp["boxes"][0], lin[0]], ["Simulations", "Theory"])
        plt.show()
conn.close()
