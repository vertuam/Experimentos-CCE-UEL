import random
from datetime import datetime

number_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Original list:", number_list)


begin = number_list[0:3]
middle = number_list[3:-2]
end = number_list[-2:]

print("begin:", begin)
print("middle:", middle)
print("end:", end)


random.seed(4) # init random
random.shuffle(middle)

out = begin + middle + end
print("After:", out)

a = ("John", "Charles", "Mike")
b = ("Jenny", "Christy", "Monica", "Vicky")

x = zip(a, b)

for y in x:
    print(y)

print(datetime.timestamp(datetime.now()))
print(datetime.fromtimestamp(datetime.timestamp(datetime.now())))
