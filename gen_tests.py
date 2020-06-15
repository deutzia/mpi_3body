#!/usr/bin/env python3.6
import random
import sys

n = int(sys.argv[1])

for i in range(n):
    print(f"{random.uniform(0, n)} {random.uniform(0, n)} {random.uniform(0, n)} {random.uniform(0, n)} {random.uniform(0, n)} {random.uniform(0, n)}")
