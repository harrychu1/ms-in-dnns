import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--length", help="length of sequence", type=int, default=10)
parser.add_argument("--sequence", help="name of sequence (fibonacci, prime, square, triangular, factorial)", default="fibonacci")

args = parser.parse_args()

def fibonacci(x):
    start = [0, 1]
    if x < 2:
        return start[:x]
    a=0
    b=1
    for i in range(x-2):
        a, b = b, a+b
        start.append(b)
    return start

def prime(x):
    start = [2]
    count=1
    p=2
    while count < x:
        p+=1
        prime=True
        for i in range(2,int(math.sqrt(p))+1):
            if p%i == 0:
                prime=False
                break
        if prime==True:
            count+=1
            start.append(p)
    return start

def square(x):
    start=[]
    for i in range(x):
        start.append((i+1)**2)
    return start

def triangular(x):
    start=[]
    for i in range(1,x+1):
        start.append(int((i*(i+1))/2))
    return start

def factorial(x):
    start=[]
    for i in range(1,x+1):
        p = 1
        for j in range(1,i+1):
            p=p*j
        start.append(p)
    return start

def main(args):
    length=args.length
    sequence=args.sequence

    if length <= 0:
        parser.error("Invalid length")

    if sequence not in ["fibonacci", "prime", "square", "triangular", "factorial"]:
        parser.error(
            "invalid choice"
        )
    if sequence == "fibonacci":
        return fibonacci(length)
    if sequence == "prime":
        return prime(length)
    if sequence == "square":
        return square(length)
    if sequence == "triangular":
        return triangular(length)
    if sequence == "factorial":
        return factorial(length)

if __name__ == "__main__":
    main(args)