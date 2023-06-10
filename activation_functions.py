import numpy as np
import matplotlib.pyplot as plt
from Finals.plot import set_ax_info
import seaborn as sns

def plot_function(name, f, color, ylim):
    x = np.linspace(-4, 4, 100)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, f(x), color)
    set_ax_info(ax, 'Input', 'Output', f'{name}, $f(x)$', legend=False)
    ax.set_ylim(ylim)
    plt.tight_layout()
    fig.savefig(f'Latex/figures/{name}.pdf')
    plt.close(fig)

# Get the seaborn color palette
blue, green, red, *_ = sns.color_palette().as_hex()

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

plot_function('Sigmoid', sigmoid, blue, (-0.25,1.25))
plot_function('Tanh', tanh, green, (-1.5,1.5))
plot_function('ReLU', relu, red, (-0.5,3))