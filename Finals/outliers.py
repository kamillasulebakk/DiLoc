import numpy as np
import matplotlib.pyplot as plt


data = np.load('data/simple_70000_1_eeg_train-validation.npy')
data = data.flatten()

mean = np.mean(data)
std = np.std(data)

threshold = 10
outliers = []
for x in data:
    z_score = (x - mean) / std
    if abs(z_score) > threshold:
        outliers.append(x)
print("Mean: ",mean)
print("\nStandard deviation: ",std)
# print("\nOutliers  : ", outliers)
print(len(data))
print(len(outliers))

# Create a scatter plot
plt.scatter(range(len(data)), data, color='b', label='Data Points')

# Plot the mean and threshold lines
plt.axhline(mean, color='r', linestyle='dashed', linewidth=2, label='Mean')
plt.axhline(mean + threshold * std, color='g', linestyle='dotted', linewidth=2, label='Upper Threshold')
plt.axhline(mean - threshold * std, color='g', linestyle='dotted', linewidth=2, label='Lower Threshold')

# Add labels and legend
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.title('Data Visualization')
plt.show()