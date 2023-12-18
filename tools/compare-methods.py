import argparse
import os
from math import inf
import numpy as np
import random
import time

"""
Instructions:
- log_size: the size will be 2^log_size
- num_tests: number of tests that will be performed
- method1: the baseline method (classic, recursive, iterative, numpy)
- method2: the efficient method (classic, recursive, iterative, numpy)
Results will be available in output_file.
"""

output_file = "comparison.csv" 
temp_execution_file = "temp.txt"
max_num_threads = 8


#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('log_size')        
parser.add_argument('num_tests')   
parser.add_argument('method1')   
parser.add_argument('method2')       
args = parser.parse_args()
log_size = int(args.log_size)
num_tests = int(args.num_tests)
method1 = args.method1
method2 = args.method2
methods = [method1, method2]
size = 2**log_size



#define a function to parse temp_execution_file and return the elapsed time
def parse_execution_time():
    with open(temp_execution_file, "r") as f:
        times_data = f.read()
    time = int(times_data.split("μs")[0])
    return time

#define a function to generate a random numpy array of complex numbers
def get_random_sequence():
    sequence = []
    for _ in range(size):
        num1 = random.randint(0,100)
        num2 = random.randint(0,100)
        sequence.append(np.cdouble((num1,num2)))
    return np.array(sequence)



#if using numpy, generate a random sequence
if "numpy" in methods:
    sequence = get_random_sequence()

#compile
os.system("cd ../build && cmake .. -DCMAKE_BUILD_TYPE=Release && make")

avg_times = [0, 0]
min_times = [inf, inf]
avg_speedup = 0

#main loop
for _ in range(num_tests):
    new_times = [0, 0]
    for i in range(2):
        method = methods[i]

        #run
        if method != "numpy":
            os.system("../build/fft " + str(size) + " timingTest " + str(max_num_threads) + " " + method + " > " + temp_execution_file)
            new_times[i] = parse_execution_time()
        else:
            t0 = time.time()
            result = np.fft.fft(sequence)
            t1 = time.time()
            new_times[i] = (t1-t0) * (10**6)
        
        #calculate times
        avg_times[i] += new_times[i]
        if new_times[i] < min_times[i]:
            min_times[i] = new_times[i]
    #calculate speedup
    avg_speedup += new_times[0] / new_times[1]

#remove temporary file
os.system("rm " + str(temp_execution_file))
            
#average times and speedups
for i in range(2):
    avg_times[i] /= num_tests
avg_speedup /= num_tests

#calculate speedups of minimum times and speedups of average times
speedup_of_avg = avg_times[0]/avg_times[1]
speedup_of_min = min_times[0]/min_times[1]

#print the results
with open(output_file, "w") as f:
    #write the csv labels
    f.write("Category,Method1,Method2\n")

    #write the results
    data = {"AvgTimes(μs)": avg_times, "MinTimes(μs)": min_times, "AvgSU": avg_speedup, "SUofAvg": speedup_of_avg, "SUofMin": speedup_of_min}
    for datum in data:
        values = data[datum]
        string = datum
        if isinstance(values, list):
            for value in values:
                string += ","+str(value)
        else:
            string += "1," + str(values)
        f.write(string + "\n")