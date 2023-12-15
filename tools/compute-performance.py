import argparse
import os
from math import inf

"""
Instructions:
- log_size: the size will be 2^log_size
- num_tests: number of tests that will be performed
Results will be available in output_file.
Make sure that the main is only generating a sequence and executing TimeEstimateFFT:
executing a DFT or recursive FFT is really slow and print other times might break the script.
"""

#parameters
output_file = "times.csv" 
temp_execution_file = "temp.txt"
threads = [1, 2, 4, 8] #changing only this will break the script: main has to be changed as well

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('log_size')        
parser.add_argument('num_tests')       
args = parser.parse_args()
log_size = int(args.log_size)
num_tests = int(args.num_tests)



#define a function to parse temp_execution_file and return a list of elapsed times
def parse_execution_times():
    with open(temp_execution_file, "r") as f:
        times_data = f.readlines()
    times = [int(line.split(" ")[-1][:-3]) for line in times_data if "Time for" in line]
    return times

#define a function to parse temp_execution_file and return a list of speedups
def parse_execution_speedups():
    with open(temp_execution_file, "r") as f:
        times_data = f.readlines()
    speedups = [float(line.split(" ")[-1][:-2]) for line in times_data if "Speedup over" in line]
    return speedups



#compile
os.system("cd ../build && cmake .. -DCMAKE_BUILD_TYPE=Release && make")

size = 2**log_size
avg_times = [0]*len(threads)
min_times = [inf]*len(threads)
avg_speedups = [0]*len(threads)

#main loop
for _ in range(num_tests):
    #run
    os.system("../build/fft " + str(size) + " > " + temp_execution_file)

    #calculate times
    new_times = parse_execution_times()
    new_speedups = parse_execution_speedups()
    for i in range(len(new_times)):
        avg_times[i] += new_times[i]
        avg_speedups[i] += new_speedups[i]
        if new_times[i] < min_times[i]:
            min_times[i] = new_times[i]

#remove temporary file
os.system("rm " + str(temp_execution_file))
            
#average times and speedups
for i in range(len(avg_times)):
    avg_times[i] /= num_tests
    avg_speedups[i] /= num_tests

#calculate speedups of minimum times and speedups of average times
speedups_of_avg = []
speedups_of_min = []
for i in range(len(avg_times)):
    speedups_of_avg.append(avg_times[0]/avg_times[i])
    speedups_of_min.append(min_times[0]/min_times[i])

#print the results
with open(output_file, "w") as f:
    #write the csv labels
    labels = "Category"
    for thread in threads:
        labels += ","+str(thread)+"Threads"
    f.write(labels + "\n")

    #write the results
    data = {"AvgTimes(μs)": avg_times, "MinTimes(μs)": min_times, "AvgSU": avg_speedups, "SUofAvg": speedups_of_avg, "SUofMin": speedups_of_min}
    for datum in data:
        values = data[datum]
        string = datum
        for value in values:
            string += ","+str(value)
        f.write(string + "\n")