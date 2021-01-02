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

parser=argparse.ArgumentParser(description="Generate Monte Carlo simulation graphs for Chapter 5 of the PhD Thesis.")

parser.add_argument("dbpath", type=str, help="Path to sqlite database")
args=parser.parse_args()

conn = sqlite3.connect(args.dbpath)
cursor = conn.cursor()

p = ["0.900000", "0.800000", "0.700000", "0.600000", "0.500000", "0.400000"]
pfloat = np.asarray(p, dtype=float)

################################################################################
#####################################GRAPHS#####################################
################################################################################

if input("Run thesis bounds: timesteps vs n? y/return: ") == "y":
    for pi in p:
        mix = 1
        data = []
        nvalues = []
        for ni in range(6, 16):
            n = 2**ni
            cursor.execute("""SELECT Timesteps FROM final WHERE n=""" + str(n) + """ and p='""" + pi + """' and alpha="0.000000" and mix=""" + str((-1*mix)))
            data.append(list(map(int,*zip(*cursor.fetchall()))))# , label="n = " + str(n))
            nvalues.append(n)
        print("p = " + str(pi) + ", mix = -1")
        fig=plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(data, showfliers=False, showcaps=False)
        ax.plot(range(1, 11), (3/float(pi)) * np.log(2) * range(6, 16))
        ax.set_xticklabels(nvalues)
        plt.ylabel("Transmission time-steps")
        plt.xlabel("n")
        plt.show()

if input("Run thesis bounds: timesteps vs p mix1? y/return: ") == "y":
    for ni in range(6, 16):
        mix = 1
        data = []
        for pi in p:
            n = 2**ni
            cursor.execute("""SELECT Timesteps FROM final WHERE n=""" + str(n) + """ and p='""" + pi + """' and alpha="0.000000" and mix=""" + str((-1*mix)))
            data.append(list(map(int,*zip(*cursor.fetchall()))))# , label="n = " + str(n))
        print("n = " + str(n) + ", mix = " + str(-mix))
        fig=plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(data, showfliers=False, showcaps=False, boxprops=dict(color=colours[mix-1]), whiskerprops=dict(color=colours[mix-1]), medianprops=dict(color=colours[mix-1]), positions=pfloat, widths=width1(pfloat))
        ax.plot(pfloat, (3*np.log(n))/(pfloat), color=colours[mix-2])
        ax.set_xscale('function', functions=(forward1, inverse1))
        plt.xlim(0.38,1)
        plt.ylabel("Transmission time-steps")
        plt.xlabel("p")
        plt.legend()
        plt.show()
conn.close()
