a = int(input("Enter the value of 'a' : "))

if(a <= 50 and a >= 0):
    if(a >= 40):
        print("value of 'a' is between 50 - 40")
    elif(a >= 30):
        print("value of 'a' is between 40 - 30")
    elif(a >= 20):
        print("value of 'a' is between 30 - 20")
    elif(a >= 10):
        print("value of 'a' is between 20 - 10")
    else:
        print("value of 'a' is between 10 - 0")
else:
    print("value out of range (0 -- 50)")

print('a = ', a)

