def add(*val):
    su=0
    for i in val:
        su+=i
    return su

print(add(10))
print(add(10, 20))
print(add(10, 20, 2.2))
