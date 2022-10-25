def print_array(a):
    if (type(a) == list):
        [print("Value of index "+str(x)+" is : "+str(y)) for x, y in enumerate(a)]

arr = [1, 1.1, "name", 'c', 2+1j]
print(arr)
print_array(arr)
