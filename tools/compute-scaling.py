import argparse
import os
from math import inf

"""
Instructions:
- min_log_size: the minimum size will be 2^min_log_size
- min_log_size: the maximum size will be 2^max_log_size
- num_tests: number of tests that will be performed for each size
Results will be available in output_file.
Make sure that the main is only generating a sequence and executing TimeEstimateFFT:
executing a DFT or recursive FFT is really slow and print other times might break the script.
"""

#parameters
output_file = "scaling.csv" 
temp_execution_file = "temp.txt"
threads = [1, 2, 4, 8] #changing only this will break the script: main has to be changed as well

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('min_log_size')  
parser.add_argument('max_log_size')       
parser.add_argument('num_tests')       
args = parser.parse_args()
min_log_size = int(args.min_log_size)
max_log_size = int(args.max_log_size)
num_tests = int(args.num_tests)



#define a function to parse temp_execution_file and return a list of elapsed times
def parse_execution_times():
    with open(temp_execution_file, "r") as f:
        times_data = f.readlines()
    times = [int(line.split(" ")[-1][:-3]) for line in times_data if "Time for" in line]
    return times



#compile
os.system("cd ../build && cmake .. -DCMAKE_BUILD_TYPE=Release && make")

#main loop
speedups = []
for log_size in range(min_log_size, max_log_size+1):
    size = 2**log_size
    min_times = [inf]*len(threads)

    #perform the needed tests
    for _ in range(num_tests):
        #run
        os.system("../build/fft " + str(size) + " scalingTest > " + temp_execution_file)

        #parse and calculate minimum times
        new_times = parse_execution_times()
        for i in range(len(new_times)):
            if new_times[i] < min_times[i]:
                min_times[i] = new_times[i]

    #calculate speedups of minimum times
    speedups_of_min = []
    for i in range(len(min_times)):
        speedups_of_min.append(min_times[0]/min_times[i])
    speedups.append(speedups_of_min)
    
        

#remove temporary file
os.system("rm " + str(temp_execution_file))

#print the results
with open(output_file, "w") as f:
    #write the csv labels
    labels = "log_2(size)"
    for thread in threads:
        labels += ","+str(thread)+"Threads"
    f.write(labels + "\n")

    #write the results
    for i, results in enumerate(speedups):
        log_size = i + min_log_size
        string = str(log_size)
        for result in results:
            string += "," + str(result)
        f.write(string + "\n")