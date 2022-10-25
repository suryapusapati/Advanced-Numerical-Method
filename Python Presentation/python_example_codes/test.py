import numpy as np
Xi = np.linspace(1,5,9)

# Check for atleast one 2.5 value in the given array.
for i in range(len(Xi)):
    if Xi[i] == 2.5:
        print("Found")
        break
    if i == len(Xi)-1:
        print("Not Found")