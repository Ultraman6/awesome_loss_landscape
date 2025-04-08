import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from scipy.stats import entropy

# 定义真实向量
true_vector1 = np.array([1, 0])
true_vector2 = np.array([0, 1])

# 采样范围
x = np.linspace(0, 1, 100)
y = 1 - x

# 初始化距离和熵值数组
distances1 = []
entropies1 = []
distances2 = []
entropies2 = []

for i in range(len(x)):
    predicted_vector = np.array([x[i], y[i]])

    # 计算与第一个真实向量的距离和熵
    distance1 = euclidean(true_vector1, predicted_vector)
    distances1.append(distance1)
    entropy1 = entropy(true_vector1, predicted_vector)
    entropies1.append(entropy1)

    # 计算与第二个真实向量的距离和熵
    distance2 = euclidean(true_vector2, predicted_vector)
    distances2.append(distance2)
    entropy2 = entropy(true_vector2, predicted_vector)
    entropies2.append(entropy2)

# 创建 3D 图形
fig = plt.figure(figsize=(12, 6))

# 绘制与第一个真实向量的距离
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x, y, distances1, label='Distance to [1, 0]')
ax1.set_xlabel('Probability of Class 1')
ax1.set_ylabel('Probability of Class 2')
ax1.set_zlabel('Distance')
ax1.set_title('Distance to [1, 0] in 3D')
ax1.legend()

# 绘制与第二个真实向量的距离
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(x, y, distances2, label='Distance to [0, 1]', color='orange')
ax2.set_xlabel('Probability of Class 1')
ax2.set_ylabel('Probability of Class 2')
ax2.set_zlabel('Distance')
ax2.set_title('Distance to [0, 1] in 3D')
ax2.legend()

plt.tight_layout()
plt.show()

# 创建新的 3D 图形用于绘制熵
fig = plt.figure(figsize=(12, 6))

# 绘制与第一个真实向量的熵
ax3 = fig.add_subplot(121, projection='3d')
ax3.plot(x, y, entropies1, label='Entropy to [1, 0]')
ax3.set_xlabel('Probability of Class 1')
ax3.set_ylabel('Probability of Class 2')
ax3.set_zlabel('Entropy')
ax3.set_title('Entropy to [1, 0] in 3D')
ax3.legend()

# 绘制与第二个真实向量的熵
ax4 = fig.add_subplot(122, projection='3d')
ax4.plot(x, y, entropies2, label='Entropy to [0, 1]', color='orange')
ax4.set_xlabel('Probability of Class 1')
ax4.set_ylabel('Probability of Class 2')
ax4.set_zlabel('Entropy')
ax4.set_title('Entropy to [0, 1] in 3D')
ax4.legend()

plt.tight_layout()
plt.show()
