nodes = [231, 512, 256, 128, 64, 32, 3]

s = 0
for i in range(1, len(nodes)):
    s += nodes[i - 1]*nodes[i]
    s += nodes[i]

print(s)
