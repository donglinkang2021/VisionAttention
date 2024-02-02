from tqdm import trange

epoch = 10000
x = 0
for i in trange(epoch):
    for j in range(epoch):
        x += 1
