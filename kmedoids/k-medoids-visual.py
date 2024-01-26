import matplotlib.pyplot as plt


X = [5, 10, 20, 30, 40, 50, 100]
K_Medoids = [122022.546093137, 111687.31791026032, 106173.22641196747, 104174.01285849532, 102222.97176033957, 100770.64215416113,
             95859.45110173624]
K_Means = [261621.002193, 129932.755006, 64472.1887616, 43005.1154519, 31525.0356871, 25404.9702966,
           12335.4581548]

plt.plot(X, K_Medoids, label = 'K Medoids',  color='blue',  linestyle='-', marker='o')
plt.plot(X, K_Means, label='K Means', color='green', linestyle='-', marker='s')

plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('K Medoids vs K Means')
plt.grid(True)

plt.legend()

plt.show()