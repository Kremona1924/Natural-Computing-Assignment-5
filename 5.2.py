from random import randint, random
from copy import deepcopy
import matplotlib.pyplot as plt

def flipbit(xm, prob):
    out = []
    for bit in xm:
        if random() <= prob:
            out.append(int(not bit))
        else:
            out.append(bit)
    return out

# We kunnen ook gewoon sum() gebruiken ipv compare_list, want de target is allemaal 1's, dus dat zou ook
# een percentage geven, maar nu kunnen we het ook voor een random bitstring gebruiken.
def compare_lists(target, xm):
    correct = 0
    for i in range(len(xm)):
        if target[i] == xm[i]:
            correct += 1
    
    return correct

string_length = 100
target = [1]*string_length

l =  100
nr_gen = 1500
prob = 1/l

og =  [randint(0, 1) for i in range(l)]
 
x_best = deepcopy(og)
bestx_score = compare_lists(target, x_best)
plot_x_best = []
for gen in range(nr_gen):
    xm = flipbit(x_best, prob)
    xm_score = compare_lists(target, xm)
    if xm_score > bestx_score:
        x_best = deepcopy(xm)
        bestx_score = xm_score
    plot_x_best.append(bestx_score)

x_replace = deepcopy(og)
replacex_score = compare_lists(target, x_replace)
plot_x_replace = []
for gen in range(nr_gen):
    xm = flipbit(x_replace, prob)
    xm_score = compare_lists(target, xm)
    x_replace = deepcopy(xm)
    replacex_score = xm_score
    plot_x_replace.append(replacex_score)

plt.figure()
plt.plot(plot_x_best, color='blue', label='use_best')
plt.plot(plot_x_replace, color='red', label='always_replace')
plt.xlim(0, nr_gen)
plt.ylim(40, string_length+1)
plt.legend()
plt.show()

