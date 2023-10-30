import numpy as np
import matplotlib.pyplot as plt
from plot import set_ax_info
import seaborn as sns

# Get the seaborn color palette
palette = sns.color_palette("deep")
sns.set_palette(palette)

plt.style.use('seaborn')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "font.serif": ["Computer Modern"]}
)

def plot_function(name, f, color, ylim):
    x = np.linspace(-4, 4, 100)
    fig = plt.figure()
    ax = fig.add_subplot()

    # Plot the activation function
    ax.plot(x, f(x), color = color, label=r' $a =$ Sigmoid$(x)$')

    # Plot the derivative with stippled lines
    # ax.plot(x, df(x), color, linestyle=':', label='$f\'(x)$')

    set_ax_info(ax, r'Input, $x$', r'Output, $a$', f'{name}', legend=True)
    ax.set_ylim(ylim)
    plt.tight_layout()
    fig.savefig(f'../Latex_new/figures/{name}.pdf')
    plt.close(fig)


# Define the activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     sig = sigmoid(x)
#     return sig * (1 - sig)

def tanh(x):
    return np.tanh(x)

# def tanh_derivative(x):
#     return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

# def relu_derivative(x):
#     return np.where(x > 0, 1, 0)

# plot_function('Sigmoid', sigmoid, sigmoid_derivative, blue, (-0.25, 1.25))
# plot_function('Tanh', tanh, tanh_derivative, green, (-1.5, 1.5))
# plot_function('ReLU', relu, relu_derivative, red, (-0.5, 3))

plot_function('Sigmoid', sigmoid, palette[0], (-0.25, 1.25))
# plot_function('Tanh', tanh, palette[2], (-1.5, 1.5))
# plot_function('ReLU', relu, palette[3], (-0.5, 4))