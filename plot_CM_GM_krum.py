import numpy as np

np.random.seed(19680801)
data = np.random.randn(2, 100)

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from matplotlib import rc
from scipy.spatial.distance import cdist, euclidean

from centerpoint.Centerpoint import *

rc('text', usetex=True)


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

plt.style.available

# 设置默认绘图风格
plt.style.use("bmh")
plt.rcParams.update({'figure.facecolor': "white",
                     "axes.facecolor" : "white"})
# plt.style.use("seaborn-white")


# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.labelsize'] = 10
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titlesize'] = 10
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
# plt.rcParams['legend.fontsize'] = 10
# plt.rcParams['figure.titlesize'] = 12


# with plt.style.context(['science']):
fig = plt.figure(figsize=(4.5,4.5))
ax = Axes3D(fig)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# Bonus: To get rid of the grid as well:
# ax.grid(False)

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].

# points = np.array(
#     [[-10, -9, 1], [-9, -5, 2], [-6, -9, 1], [2, 6, 8], [2, 8, 7], [3, 7, 9], [6, 5, 7.5], [7, 4, 8.5], [10, 2, 9.5],
#      [9, 3, 10]]) / 10
# points = np.array(
#     [ [1, 1, 2], [1, 0.5, 1], [0, 0, 1],[2, 6, 8], [2, 8, 7], [3, 7, 9], [6, 5, 7.5], [7, 4, 8.5], [10, 2, 9.5],
#      [9, 3, 10]]) / 10
# points = np.array(
#     [ [1, 1, 2], [1, 0.5, 1], [2, 6, 8], [2, 8, 7], [3, 7, 9], [6, 5, 7.5], [7, 4, 8.5], [10, 2, 9.5],
#      [9, 3, 10]]) / 10
# points = np.array(
#     [ [5, 5, 6], [6, 6, 5], [2, 2, 8], [2, 3, 7], [3, 7, 9], [6, 5, 7.5], [10, 10, 8.5], [10, 2, 9.5],
#      [9, 3, 10]]) / 10
points = np.array(
    [ [9,9, 3.5], [8,8,4], [1,1, 3], [1,5, 4], [4,4, 5], [6, 10, 10], [10, 10, 8.5], [10, 2, 9.5],
     [9, 5, 10]]) / 10
normal_points = points[2:]
bad_points = points[:2]

n=9
f = (n-2)//2

krum = points[-1]
min_sum_dist = np.inf
for p in points:
    dist = []
    for neighbor in points:
        dist.append(np.linalg.norm(p - neighbor)**2)
    dist = np.sort(dist)
    sum_dist = sum(dist[:n-f])
    print(sum_dist)
    if sum_dist < min_sum_dist:
        min_sum_dist = sum_dist
        krum = p


# with plt.style.context(['science']):
xs = [x[0] for x in bad_points]
ys = [x[1] for x in bad_points]
zs = [x[2] for x in bad_points]
ax.plot(xs, ys, zs, 'o', markersize=5, color='r', label='Byzantine')


# get and plot hull
hull = ConvexHull(normal_points)
vertices = [normal_points[s] for s in hull.simplices]
triangles = Poly3DCollection(vertices, edgecolor='b', facecolor='b', linewidths=0.2, alpha=0.2)
ax.add_collection3d(triangles)

xs = [x[0] for x in normal_points]
ys = [x[1] for x in normal_points]
zs = [x[2] for x in normal_points]
ax.plot(xs, ys, zs, 'o', markersize=5, color='b', label='Normal')

average = np.mean(np.concatenate((normal_points, bad_points), axis=0), axis=0)
CM = np.median(np.concatenate((normal_points, bad_points), axis=0), axis=0)
GM = geometric_median(np.concatenate((normal_points, bad_points), axis=0))
print(average, CM, GM, krum)

# ax.scatter(average[0], average[1], average[2], color='m', label='Average',  marker='^', s=30)
ax.scatter(CM[0], CM[1], CM[2], color='g', label='Coordinate-wise Median', marker='^', s=30)
ax.scatter(GM[0], GM[1], GM[2], color='orange', label='Geometric Median',  marker='P', s=30)
ax.scatter(krum[0], krum[1], krum[2], color='y', label='Krum',  marker='*', s=30)


# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([0,1])
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1])

plt.legend(loc='best')
ax.legend(bbox_to_anchor=(0.3, 0.45))
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig("plot_average_CM_GM_3D.png", bbox_inches='tight')
plt.show()
