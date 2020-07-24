import copy

from matplotlib import rc
from scipy.spatial.distance import cdist, euclidean

from centerpoint.Centerpoint import *
from utils import *

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def noncooperativeMobileNet(i, numAgents, w0, x, vu, u, vd, d, sensingRange, psi, w, mu_k, q, phi, vg, niu_k, v, lamda,
                            beta, gamma, r, delta, attackers, psi_a, phi_a):
    for k in range(numAgents):
        dist = w0.distance(x[k])
        unit = [(w0.x - x[k].x) / dist, (w0.y - x[k].y) / dist]
        if dist > 2:
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
        else:
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
            # u[:, k] = unit + vu[i, k] * (dist + 1) / 10
            # d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k] * (dist + 1) / 10

        q[:, k] = [x[k].x, x[k].y] + d[k] * u[:, k]
    a = 0
    for k in range(numAgents):
        if k not in attackers:
            # target estimation
            psi[:, k] = w[:, k] + mu_k * (q[:, k] - w[:, k])
            w[:, k] = psi[:, k]
            # velocity estimation
            phi[:, k] = vg[:, k] + niu_k * (v[:, k] - vg[:, k])
            vg[:, k] = phi[:, k]
        else:
            w[:, k] = psi_a[:, a]
            vg[:, k] = phi_a[:, a]
            a += 1

    # update node velocity and its location
    for k in range(numAgents):
        if k not in attackers:
            v[:, k] = lamda * h(Point(w[0, k], w[1, k]), x[k], 1) + beta * vg[:, k] + gamma * Delta(x, k, numAgents,
                                                                                                    r=r,
                                                                                                    sensingRange=sensingRange)
            x[k] = Point(x[k].x + delta * v[0, k], x[k].y + delta * v[1, k])
        else:
            # v[:, k] = vg[:, k]
            # x[k] = Point(x[k].x + delta * v[0, k], x[k].y + delta * v[1, k])
            v[:, k] = vg[:, k]
            x[k] = x[k]

    return x, w, v, vg


def averageMobileNet(i, numAgents, w0, x, vu, u, vd, d, sensingRange, psi, w, mu_k, q, phi, vg, niu_k, v, lamda,
                     beta, gamma, r, delta, attackers, psi_a, phi_a):
    for k in range(numAgents):
        dist = w0.distance(x[k])
        unit = [(w0.x - x[k].x) / dist, (w0.y - x[k].y) / dist]
        if dist > 2:
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
        else:
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
            # u[:, k] = unit + vu[i, k] * (dist + 1) / 10
            # d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k] * (dist + 1) / 10
        q[:, k] = [x[k].x, x[k].y] + d[k] * u[:, k]

    Neigh = []
    a = 0
    for k in range(numAgents):
        if k not in attackers:
            neighbor = findNeighbors(x, k, numAgents, sensingRange, completeGraph=True)
            Neigh.append(neighbor)
            neighborSize = len(neighbor)
            Cmatrix = np.zeros((numAgents, 1))
            Amatrix = np.zeros((numAgents, 1))
            for n in neighbor:
                if n == k:
                    Cmatrix[n, 0] = 1
                Amatrix[n, 0] = 1 / neighborSize
            # target estimation
            psi[:, k] = w[:, k] + mu_k * np.dot((q - (w[:, k])[:, np.newaxis]), Cmatrix).squeeze()

            # velocity estimation
            phi[:, k] = vg[:, k] + niu_k * np.dot((v - (vg[:, k])[:, np.newaxis]), Cmatrix).squeeze()
        else:
            Neigh.append([])
            psi[:, k] = psi_a[:, a]
            phi[:, k] = phi_a[:, a]
            a += 1

    for k in range(numAgents):
        if k not in attackers:
            w[:, k] = np.mean(np.array([psi[:, j] for j in Neigh[k]]), axis=0)
            vg[:, k] = np.mean(np.array([phi[:, j] for j in Neigh[k]]), axis=0)
        else:
            w[:, k] = psi[:, k]
            vg[:, k] = phi[:, k]

        # update node velocity and its location
    for k in range(numAgents):
        if k not in attackers:
            v[:, k] = lamda * h(Point(w[0, k], w[1, k]), x[k], 1) + beta * vg[:, k] + gamma * Delta(x, k, numAgents,
                                                                                                    r=r,
                                                                                                    sensingRange=sensingRange)
            x[k] = Point(x[k].x + delta * v[0, k], x[k].y + delta * v[1, k])
        else:
            # v[:, k] = phi[:, k]
            # x[k] = Point(x[k].x + delta * v[0, k], x[k].y + delta * v[1, k])
            v[:, k] = vg[:, k]
            x[k] = x[k]

    return x, w, v, vg


def GMMobileNet(i, numAgents, w0, x, vu, u, vd, d, sensingRange, psi, w, mu_k, q, phi, vg, niu_k, v, lamda,
                beta, gamma, r, delta, attackers, psi_a, phi_a):
    for k in range(numAgents):
        dist = w0.distance(x[k])
        unit = [(w0.x - x[k].x) / dist, (w0.y - x[k].y) / dist]
        if dist > 2:
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
        else:
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
            # u[:, k] = unit + vu[i, k] * (dist + 1) / 10
            # d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k] * (dist + 1) / 10
        q[:, k] = [x[k].x, x[k].y] + d[k] * u[:, k]

    Neigh = []
    a = 0
    for k in range(numAgents):
        if k not in attackers:
            neighbor = findNeighbors(x, k, numAgents, sensingRange, completeGraph=True)
            Neigh.append(neighbor)
            neighborSize = len(neighbor)
            Cmatrix = np.zeros((numAgents, 1))
            Amatrix = np.zeros((numAgents, 1))
            for n in neighbor:
                if n == k:
                    Cmatrix[n, 0] = 1
                Amatrix[n, 0] = 1 / neighborSize
            # target estimation
            psi[:, k] = w[:, k] + mu_k * np.dot((q - (w[:, k])[:, np.newaxis]), Cmatrix).squeeze()

            # velocity estimation
            phi[:, k] = vg[:, k] + niu_k * np.dot((v - (vg[:, k])[:, np.newaxis]), Cmatrix).squeeze()
        else:
            Neigh.append([])
            psi[:, k] = psi_a[:, a]
            phi[:, k] = phi_a[:, a]
            a += 1

    for k in range(numAgents):
        if k not in attackers:
            w[:, k] = geometric_median(np.array([psi[:, j] for j in Neigh[k]]))
            vg[:, k] = geometric_median(np.array([phi[:, j] for j in Neigh[k]]))
        else:
            w[:, k] = psi[:, k]
            vg[:, k] = phi[:, k]

        # update node velocity and its location
    for k in range(numAgents):
        if k not in attackers:
            v[:, k] = lamda * h(Point(w[0, k], w[1, k]), x[k], 1) + beta * vg[:, k] + gamma * Delta(x, k, numAgents,
                                                                                                    r=r,
                                                                                                    sensingRange=sensingRange)
            x[k] = Point(x[k].x + delta * v[0, k], x[k].y + delta * v[1, k])
        else:
            # v[:, k] = phi[:, k]
            # x[k] = Point(x[k].x + delta * v[0, k], x[k].y + delta * v[1, k])
            v[:, k] = vg[:, k]
            x[k] = x[k]

    return x, w, v, vg


def medianMobileNet(i, numAgents, w0, x, vu, u, vd, d, sensingRange, psi, w, mu_k, q, phi, vg, niu_k, v, lamda,
                    beta, gamma, r, delta, attackers, psi_a, phi_a):
    for k in range(numAgents):
        dist = w0.distance(x[k])
        unit = [(w0.x - x[k].x) / dist, (w0.y - x[k].y) / dist]
        if dist > 2:
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
        else:
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
            # u[:, k] = unit + vu[i, k] * (dist + 1) / 10
            # d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k] * (dist + 1) / 10
        q[:, k] = [x[k].x, x[k].y] + d[k] * u[:, k]

    Neigh = []
    a = 0
    for k in range(numAgents):
        if k not in attackers:
            neighbor = findNeighbors(x, k, numAgents, sensingRange, completeGraph=True)
            Neigh.append(neighbor)
            neighborSize = len(neighbor)
            Cmatrix = np.zeros((numAgents, 1))
            Amatrix = np.zeros((numAgents, 1))
            for n in neighbor:
                if n == k:
                    Cmatrix[n, 0] = 1
                Amatrix[n, 0] = 1 / neighborSize
            # target estimation
            psi[:, k] = w[:, k] + mu_k * np.dot((q - (w[:, k])[:, np.newaxis]), Cmatrix).squeeze()

            # velocity estimation
            phi[:, k] = vg[:, k] + niu_k * np.dot((v - (vg[:, k])[:, np.newaxis]), Cmatrix).squeeze()
        else:
            Neigh.append([])
            psi[:, k] = psi_a[:, a]
            phi[:, k] = phi_a[:, a]
            a += 1

    for k in range(numAgents):
        if k not in attackers:
            w[:, k] = np.median(np.array([psi[:, j] for j in Neigh[k]]), axis=0)
            vg[:, k] = np.median(np.array([phi[:, j] for j in Neigh[k]]), axis=0)
        else:
            w[:, k] = psi[:, k]
            vg[:, k] = phi[:, k]

        # update node velocity and its location
    for k in range(numAgents):
        if k not in attackers:
            v[:, k] = lamda * h(Point(w[0, k], w[1, k]), x[k], 1) + beta * vg[:, k] + gamma * Delta(x, k, numAgents,
                                                                                                    r=r,
                                                                                                    sensingRange=sensingRange)
            x[k] = Point(x[k].x + delta * v[0, k], x[k].y + delta * v[1, k])
        else:
            # v[:, k] = phi[:, k]
            # x[k] = Point(x[k].x + delta * v[0, k], x[k].y + delta * v[1, k])
            v[:, k] = vg[:, k]
            x[k] = x[k]

    return x, w, v, vg


def centerpointMobileNet(i, numAgents, w0, x, vu, u, vd, d, sensingRange, psi, w, mu_k, q, phi, vg, niu_k, v, lamda,
                         beta, gamma, r, delta, attackers, psi_a, phi_a):
    for k in range(numAgents):
        dist = w0.distance(x[k])
        unit = [(w0.x - x[k].x) / dist, (w0.y - x[k].y) / dist]
        if dist > 2:
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
        else:
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
            # u[:, k] = unit + vu[i, k] * (dist + 1) / 10
            # d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k] * (dist + 1) / 10
        q[:, k] = [x[k].x, x[k].y] + d[k] * u[:, k]

    Neigh = []
    a = 0
    for k in range(numAgents):
        if k not in attackers:
            neighbor = findNeighbors(x, k, numAgents, sensingRange, completeGraph=True)
            Neigh.append(neighbor)
            neighborSize = len(neighbor)
            Cmatrix = np.zeros((numAgents, 1))
            Amatrix = np.zeros((numAgents, 1))
            for n in neighbor:
                if n == k:
                    Cmatrix[n, 0] = 1
                Amatrix[n, 0] = 1 / neighborSize
            # target estimation
            psi[:, k] = w[:, k] + mu_k * np.dot((q - (w[:, k])[:, np.newaxis]), Cmatrix).squeeze()

            # velocity estimation
            phi[:, k] = vg[:, k] + niu_k * np.dot((v - (vg[:, k])[:, np.newaxis]), Cmatrix).squeeze()
        else:
            Neigh.append([])
            psi[:, k] = psi_a[:, a]
            phi[:, k] = phi_a[:, a]
            a += 1

    for k in range(numAgents):
        if k not in attackers:
            # try:
            #     w[:, k] = [cp_w.x, cp_w.y]
            #     vg[:, k] = [cp_vg.x, cp_vg.y]
            # except:
            centerPoint = Centerpoint()
            try:
                cp_w = centerPoint.getSafeCenterPoint([Point(psi[0, j], psi[1, j]) for j in Neigh[k]])
            except:
                cp_w = Point(psi[0, 10], psi[1, 10])
            w[:, k] = [cp_w.x, cp_w.y]
            try:
                cp_vg = centerPoint.getSafeCenterPoint([Point(phi[0, j], phi[1, j]) for j in Neigh[k]])
            except:
                cp_vg = Point(phi[0, 10], phi[1, 10])
            vg[:, k] = [cp_vg.x, cp_vg.y]
        else:
            w[:, k] = psi[:, k]
            vg[:, k] = phi[:, k]

        # update node velocity and its location
    print(w[:, 0], vg[:, 0])
    for k in range(numAgents):
        if k not in attackers:
            v[:, k] = lamda * h(Point(w[0, k], w[1, k]), x[k], 1) + beta * vg[:, k] + gamma * Delta(x, k, numAgents,
                                                                                                    r=r,
                                                                                                    sensingRange=sensingRange)
            x[k] = Point(x[k].x + delta * v[0, k], x[k].y + delta * v[1, k])
        else:
            # v[:, k] = phi[:, k]
            # x[k] = Point(x[k].x + delta * v[0, k], x[k].y + delta * v[1, k])
            v[:, k] = vg[:, k]
            x[k] = x[k]

    return x, w, v, vg


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


def plotDynamics():
    plt.clf()

    ax1 = plt.subplot(151)
    ax1.set_xlim(-0.5, box)
    ax1.set_ylim(-0.5, box)
    ax1.set_aspect('equal', adjustable='box')
    for tic in ax1.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax1.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    plot_point(w0, marker='*', color='y', size=12, ax=ax1)
    plot_point_set(x_no, color='b', ax=ax1, alpha=0.5)
    if attackers:
        for i in attackers:
            plot_point(x_no[i], color='r', ax=ax1)
    draw_circle = plt.Circle((5, 5), 1, fill=False, color='y', linestyle='dashed')
    ax1.add_artist(draw_circle)
    ax1.set_title('Noncooperative SGD')

    ax5 = plt.subplot(152)
    ax5.set_xlim(-0.5, box)
    ax5.set_ylim(-0.5, box)
    ax5.set_aspect('equal', adjustable='box')
    for tic in ax5.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax5.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    plot_point(w0, marker='*', color='y', size=12, ax=ax5)
    plot_point_set(x_avg, color='b', ax=ax5, alpha=0.5)
    draw_circle = plt.Circle((5, 5), 1, fill=False, color='y', linestyle='dashed')
    ax5.add_artist(draw_circle)
    if attackers:
        for i in attackers:
            plot_point(x_avg[i], color='r', ax=ax5)
    ax5.set_title('Average')

    ax2 = plt.subplot(153)
    ax2.set_xlim(-0.5, box)
    ax2.set_ylim(-0.5, box)
    ax2.set_aspect('equal', adjustable='box')
    for tic in ax2.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax2.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    plot_point(w0, marker='*', color='y', size=12, ax=ax2)
    plot_point_set(x_co, color='b', ax=ax2, alpha=0.5)
    draw_circle = plt.Circle((5, 5), 1, fill=False, color='y', linestyle='dashed')
    ax2.add_artist(draw_circle)
    if attackers:
        for i in attackers:
            plot_point(x_co[i], color='r', ax=ax2)
    ax2.set_title('Coordinate-wise median')

    ax3 = plt.subplot(154)
    ax3.set_xlim(-0.5, box)
    ax3.set_ylim(-0.5, box)
    ax3.set_aspect('equal', adjustable='box')
    for tic in ax3.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax3.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    plot_point(w0, marker='*', color='y', size=12, ax=ax3)
    plot_point_set(x_GM, color='b', ax=ax3, alpha=0.5)
    draw_circle = plt.Circle((5, 5), 1, fill=False, color='y', linestyle='dashed')
    ax3.add_artist(draw_circle)
    if attackers:
        for i in attackers:
            plot_point(x_GM[i], color='r', ax=ax3)
    ax3.set_title('Geometric median')

    ax4 = plt.subplot(155)
    ax4.set_xlim(-0.5, box)
    ax4.set_ylim(-0.5, box)
    ax4.set_aspect('equal', adjustable='box')
    for tic in ax4.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax4.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    plot_point(w0, marker='*', color='y', size=12, ax=ax4)
    plot_point_set(x_center, color='b', ax=ax4, alpha=0.5)
    draw_circle = plt.Circle((5, 5), 1, fill=False, color='y', linestyle='dashed')
    ax4.add_artist(draw_circle)
    if attackers:
        for i in attackers:
            plot_point(x_center[i], color='r', ax=ax4)
    ax4.set_title('Centerpoint')

    plt.pause(0.001)


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    # ATTACK: 0 or 6 attackers as used in the paper.
    attackerNum = 6  # 0
    # parameters
    iteration = 1000
    # NO ATTACK PARAMETERS
    # delta = 0.5
    # beta = 0.5
    # lamda = 0.5
    # gamma = 0.02
    # ATTACK PARAMETERS
    # delta = 0.5
    # beta = 0.3
    # lamda = 0.8
    # gamma = 0.01
    lamda = 0.5
    beta = 0.1
    # gamma = 0.01
    gamma = 0
    delta = 0.2
    sensingRange = 10
    r = 1
    box = 7
    numAgents = 20
    mu_k = 0.05
    niu_k = 0.05
    w0 = Point(5, 5)


    attackers = random.sample(list(range(numAgents)), attackerNum)
    normalAgents = [k for k in range(numAgents) if k not in attackers]

    x_no = random_point_set(numAgents, lower=0, upper=1)
    # for k in attackers:
    #     x_no[k] = Point(1*np.random.random(), 1*np.random.random())

    x_init = copy.deepcopy(x_no)
    x_avg = copy.deepcopy(x_no)
    x_co = copy.deepcopy(x_no)
    x_center = copy.deepcopy(x_no)
    x_GM = copy.deepcopy(x_no)

    psi_a = 0 * np.ones((2, len(attackers)))
    phi_a = 0 * np.ones((2, len(attackers)))

    mu_vd = 0
    mu_vu = 0
    sigma_vd2 = 0.5 + 0.5 * np.random.random((numAgents, 1))
    sigma_vd2[random.sample(range(numAgents), 20)] = 5
    sigma_vu2 = 0 + 0.5 * np.random.random((numAgents, 1))
    sigma_vu2[random.sample(range(numAgents), 5)] = 1

    # The following parameters work
    # sigma_vd2 = 1 + 0.4 * np.random.random((numAgents, 1))
    # #sigma_vd2[random.sample(range(numAgents), 3)] = 3
    # sigma_vu2 = 0.5 + 0.05 * np.random.random((numAgents, 1))
    # #sigma_vu2[random.sample(range(numAgents), 3)] = 3
    vd = np.zeros((iteration, numAgents))
    vu = np.zeros((iteration, numAgents))
    for k in range(numAgents):
        vd[:, k] = np.random.normal(mu_vd, sigma_vd2[k], iteration)
        vu[:, k] = np.random.normal(mu_vu, sigma_vu2[k], iteration)

    d = np.zeros((numAgents,))
    u = np.zeros((2, numAgents))
    q = np.zeros((2, numAgents))
    psi = np.zeros((2, numAgents))
    w_no = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_no[0, k], w_no[1, k] = x_no[k].x, x_no[k].y
    w_avg = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_avg[0, k], w_avg[1, k] = x_avg[k].x, x_avg[k].y
    w_co = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_co[0, k], w_co[1, k] = x_co[k].x, x_co[k].y
    w_center = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_center[0, k], w_center[1, k] = x_center[k].x, x_center[k].y
    w_GM = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_GM[0, k], w_GM[1, k] = x_GM[k].x, x_GM[k].y

    vg_no = np.zeros((2, numAgents))
    vg_avg = np.zeros((2, numAgents))
    vg_co = np.zeros((2, numAgents))
    vg_center = np.zeros((2, numAgents))
    vg_GM = np.zeros((2, numAgents))
    phi = np.zeros((2, numAgents))
    v_no = np.zeros((2, numAgents))
    v_avg = np.zeros((2, numAgents))
    v_co = np.zeros((2, numAgents))
    v_center = np.zeros((2, numAgents))
    v_GM = np.zeros((2, numAgents))
    MSE_x_no = np.zeros((iteration,))
    minSE_x_no = np.zeros((iteration,))
    maxSE_x_no = np.zeros((iteration,))
    MSE_x_avg = np.zeros((iteration,))
    MSE_x_co = np.zeros((iteration,))
    MSE_x_center = np.zeros((iteration,))
    MSE_x_GM = np.zeros((iteration,))

    W1_no = np.zeros((iteration, numAgents))
    W1_center = np.zeros((iteration, numAgents))
    W1_CM = np.zeros((iteration, numAgents))
    W1_GM = np.zeros((iteration, numAgents))
    W1_avg = np.zeros((iteration, numAgents))

    fig = plt.figure(figsize=(15, 4))
    ax1 = plt.subplot(151)
    ax2 = plt.subplot(153)
    ax3 = plt.subplot(154)
    ax4 = plt.subplot(155)
    ax5 = plt.subplot(152)

    for i in range(iteration):

        w0 = Point(5+math.sin(i*0.01), 5+math.cos(i*0.01))

        error_no = 0
        error_no_min = np.inf
        error_no_max = -np.inf
        for k in normalAgents:
            agent = Point(w_no[0, k], w_no[1, k])
            e = agent.distance(w0) ** 2
            error_no_max = e if e > error_no_max else error_no_max
            error_no_min = e if e < error_no_min else error_no_min
            error_no += e
            W1_no[i, k] = w_no[0, k]
        error_no /= numAgents - attackerNum
        MSE_x_no[i] = error_no
        minSE_x_no[i] = error_no_min
        maxSE_x_no[i] = error_no_max


        error_avg = 0
        for k in normalAgents:
            agent = Point(w_avg[0, k], w_avg[1, k])
            error_avg += agent.distance(w0) ** 2
            W1_avg[i, k] = w_avg[0, k]
        error_avg /= numAgents - attackerNum
        MSE_x_avg[i] = error_avg

        error_center = 0
        for k in normalAgents:
            agent = Point(w_center[0, k], w_center[1, k])
            error_center += (agent.distance(w0)) ** 2
            W1_center[i, k] = w_center[0, k]
        error_center /= numAgents - attackerNum
        MSE_x_center[i] = error_center

        error_co = 0
        for k in normalAgents:
            agent = Point(w_co[0, k], w_co[1, k])
            error_co += (agent.distance(w0)) ** 2
            W1_CM[i, k] = w_co[0, k]
        error_co /= numAgents - attackerNum
        MSE_x_co[i] = error_co

        error_GM = 0
        for k in normalAgents:
            agent = Point(w_GM[0, k], w_GM[1, k])
            error_GM += agent.distance(w0) ** 2
            W1_GM[i, k] = w_GM[0, k]
        error_GM /= numAgents - attackerNum
        MSE_x_GM[i] = error_GM

        print('iteration %d' % i)

        plotDynamics()

        # noncooperative
        x_no, w_no, v_no, vg_no = noncooperativeMobileNet(i, numAgents, w0, x_no, vu, u, vd, d, sensingRange, psi, w_no,
                                                          mu_k, q, phi, vg_no,
                                                          niu_k, v_no, lamda, beta, gamma, r, delta, attackers, psi_a,
                                                          phi_a)

        # cooperative
        x_avg, w_avg, v_avg, vg_avg = averageMobileNet(i, numAgents, w0, x_avg, vu, u, vd, d, sensingRange, psi, w_avg,
                                                       mu_k, q, phi, vg_avg,
                                                       niu_k, v_avg, lamda, beta, gamma, r, delta, attackers, psi_a,
                                                       phi_a)

        x_co, w_co, v_co, vg_co = medianMobileNet(i, numAgents, w0, x_co, vu, u, vd, d, sensingRange, psi, w_co,
                                                  mu_k, q, phi, vg_co,
                                                  niu_k, v_co, lamda, beta, gamma, r, delta, attackers, psi_a, phi_a)

        x_GM, w_GM, v_GM, vg_GM = GMMobileNet(i, numAgents, w0, x_GM, vu, u, vd, d, sensingRange, psi, w_GM,
                                              mu_k, q, phi, vg_GM,
                                              niu_k, v_GM, lamda, beta, gamma, r, delta, attackers, psi_a, phi_a)

        # centerpoint
        x_center, w_center, v_center, vg_center = centerpointMobileNet(i, numAgents, w0, x_center, vu, u, vd, d,
                                                                       sensingRange, psi, w_center,
                                                                       mu_k, q, phi, vg_center,
                                                                       niu_k, v_center, lamda, beta, gamma, r, delta,
                                                                       attackers, psi_a, phi_a)

        # a = 0
        # for k in range(numAgents):
        #     if k in attackers:
        #         phi_a[:, a] = v_center[:, k]
        #         psi_a[:, a] = [x_center[k].x, x_center[k].y]
        #         a += 1

    fig0 = plt.figure(figsize=(2.5, 2.5))
    ax00 = plt.axes()
    ax00.set_xlim(-0.5, box)
    ax00.set_ylim(-0.5, box)
    ax00.set_aspect('equal', adjustable='box')
    for tic in ax00.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax00.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    if attackers:
        for i in attackers:
            plot_point(x_init[i], color='r', ax=ax00)
    plot_point(w0, marker='*', color='y', size=12, ax=ax00)
    plot_point_set(x_init, color='b', ax=ax00)
    # ax1.set_title(r'$\text{(a)}$')
    fig0.savefig('fig/deployment_init_attack%d.png' % attackerNum)

    fig = plt.figure(figsize=(12.5, 2.5))
    ax1 = plt.subplot(151)
    ax2 = plt.subplot(152)
    ax3 = plt.subplot(153)
    ax4 = plt.subplot(154)
    ax5 = plt.subplot(155)

    ax0 = plt.subplot(151)
    ax0.set_xlim(-0.5, box)
    ax0.set_ylim(-0.5, box)
    ax0.set_aspect('equal', adjustable='box')
    for tic in ax0.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax0.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    if attackers:
        for i in attackers:
            plot_point(x_no[i], color='r', ax=ax0)
    draw_circle = plt.Circle((5, 5), 1, fill=False, color='y', linestyle='dashed')
    ax0.add_artist(draw_circle)
    plot_point(w0, marker='*', color='y', size=12, ax=ax0)
    plot_point_set([x_no[p] for p in normalAgents], color='b', ax=ax0)
    # ax1.set_title(r'$\text{(a)}$')

    ax1 = plt.subplot(152)
    ax1.set_xlim(-0.5, box)
    ax1.set_ylim(-0.5, box)
    ax1.set_aspect('equal', adjustable='box')
    for tic in ax1.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax1.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    if attackers:
        for i in attackers:
            plot_point(x_avg[i], color='r', ax=ax1)
    plot_point(w0, marker='*', color='y', size=12, ax=ax1)
    plot_point_set([x_avg[p] for p in normalAgents], color='b', ax=ax1)
    draw_circle = plt.Circle((5, 5), 1, fill=False, color='y', linestyle='dashed')
    ax1.add_artist(draw_circle)
    # ax1.set_title(r'$\text{(a)}$')

    ax2 = plt.subplot(153)
    ax2.set_xlim(-0.5, box)
    ax2.set_ylim(-0.5, box)
    ax2.set_aspect('equal', adjustable='box')
    for tic in ax2.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax2.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    if attackers:
        for i in attackers:
            plot_point(x_co[i], color='r', ax=ax2)
    plot_point(w0, marker='*', color='y', size=12, ax=ax2)
    plot_point_set([x_co[p] for p in normalAgents], color='b', ax=ax2)
    draw_circle = plt.Circle((5, 5), 1, fill=False, color='y', linestyle='dashed')
    ax2.add_artist(draw_circle)
    # ax2.set_title(r'(b)')

    ax3 = plt.subplot(154)
    ax3.set_xlim(-0.5, box)
    ax3.set_ylim(-0.5, box)
    ax3.set_aspect('equal', adjustable='box')
    for tic in ax3.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax3.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    if attackers:
        for i in attackers:
            plot_point(x_GM[i], color='r', ax=ax3)
    draw_circle = plt.Circle((5, 5), 1, fill=False, color='y', linestyle='dashed')
    ax3.add_artist(draw_circle)
    plot_point(w0, marker='*', color='y', size=12, ax=ax3)
    plot_point_set([x_GM[p] for p in normalAgents], color='b', ax=ax3)
    # ax3.set_title(r'(c)')

    ax4 = plt.subplot(155)
    ax4.set_xlim(-0.5, box)
    ax4.set_ylim(-0.5, box)
    ax4.set_aspect('equal', adjustable='box')
    for tic in ax4.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax4.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    if attackers:
        for i in attackers:
            plot_point(x_center[i], color='r', ax=ax4)
    plot_point(w0, marker='*', color='y', size=12, ax=ax4)
    plot_point_set([x_center[p] for p in normalAgents], color='b', ax=ax4)
    draw_circle = plt.Circle((5, 5), 1, fill=False, color='y', linestyle='dashed')
    ax4.add_artist(draw_circle)
    # ax4.set_title(r'(d)')
    fig.savefig('fig/deployment_attack%d.png' % attackerNum)

    fig1 = plt.figure(figsize=(3.9, 2.5))
    # fig1 = plt.figure(figsize=(3.9, 2))
    plt.plot(10 * np.log10(MSE_x_no[1:]), color='b', label=r'Noncoop-SGD-mean')
    plt.plot(10 * np.log10(minSE_x_no[1:]), color='b', label=r'Noncoop-SGD-min')
    plt.plot(10 * np.log10(maxSE_x_no[1:]), color='b', label=r'Noncoop-SGD-min')
    plt.plot(10 * np.log10(MSE_x_avg[1:]), label=r'Average')
    plt.plot(10 * np.log10(MSE_x_co[1:]), label=r'CM')
    plt.plot(10 * np.log10(MSE_x_GM[1:]), label=r'GM')
    plt.plot(10 * np.log10(MSE_x_center[1:]), label=r'Centerpoint')
    print("MSE_x_no=", MSE_x_no[1:])
    print("MinSE_x_no=", minSE_x_no[1:])
    print("MaxSE_x_no=", maxSE_x_no[1:])
    print("MSE_x_avg=", MSE_x_avg[1:])
    print("MSE_x_co=", MSE_x_co[1:])
    print("MSE_x_GM=", MSE_x_GM[1:])
    print("MSE_x_center=", MSE_x_center[1:])

    # plt.title('cooperative under attack using median')
    plt.xlabel(r'iteration $i$', fontsize=10)
    # plt.ylabel(r'MSD (dB)', fontsize=10)
    plt.ylabel(r'MSD', fontsize=10)
    # plt.xticks([0, 100, 200, 300, 400, 500])
    # plt.legend(fontsize=7, loc='lower left', bbox_to_anchor=(0.34, 0.43))
    # plt.yticks([-30,-15,0,15,30])
    # plt.legend(fontsize=7, loc='best')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.yticks([-75, -50, -25, 0, 25])
    # plt.yscale('log', basey=10)
    # plt.yscale('symlog')
    # if attackerNum == 6:
    #     plt.yticks([-40, -20, 0, 20, 40])
    #     plt.ylim([-45, 45])
    # elif attackerNum == 0:
    #     plt.yticks([-60, -40, -20, 0, 20])
    #     plt.ylim([-70, 40])
    plt.tight_layout()
    # plt.show()
    fig1.savefig('fig/MSD_mobile_attack%d.png' % attackerNum)

    fig = plt.figure(figsize=(15, 4))
    plt.subplot(151)
    plt.plot(10 * np.log10(MSE_x_no))
    plt.title('Noncooperative SGD')
    plt.xlabel('iteration')
    plt.ylabel('MSE (dB)')

    plt.subplot(152)
    plt.plot(10 * np.log10(MSE_x_avg))
    plt.title('Average')
    plt.xlabel('iteration')
    plt.ylabel('MSE (dB)')

    plt.subplot(153)
    plt.plot(10 * np.log10(MSE_x_co))
    plt.title('Coordinate-wise median')
    plt.xlabel('iteration')
    plt.ylabel('MSE (dB)')

    plt.subplot(154)
    plt.plot(10 * np.log10(MSE_x_GM))
    plt.title('Geometric median')
    plt.xlabel('iteration')
    plt.ylabel('MSE (dB)')

    plt.subplot(155)
    plt.plot(10 * np.log10(MSE_x_center))
    plt.title('Centerpoint')
    plt.xlabel('iteration')
    plt.ylabel('MSE (dB)')
    plt.tight_layout()
    plt.show()

    fig2 = plt.figure(figsize=(11, 2.5))
    plt.subplot(151)
    for k in normalAgents:
        plt.plot(W1_no[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.ylabel(r'$\theta_{k,i}(1)$', fontsize=25)
    plt.ylim([0,6.5])
    # plt.xticks([0, 100, 200, 300, 400, 500])

    plt.subplot(152)
    for k in normalAgents:
        plt.plot(W1_avg[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.ylim([0,6.5])
    # plt.xticks([0, 100, 200, 300, 400, 500])

    plt.subplot(153)
    for k in normalAgents:
        plt.plot(W1_CM[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.ylim([0,6.5])
    # plt.xticks([0, 100, 200, 300, 400, 500])

    plt.subplot(154)
    for k in normalAgents:
        plt.plot(W1_GM[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.ylim([0,6.5])
    # plt.xticks([0, 100, 200, 300, 400, 500])

    plt.subplot(155)
    for k in normalAgents:
        plt.plot(W1_center[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    # plt.xticks([0, 100, 200, 300, 400, 500])
    plt.tight_layout()
    plt.ylim([0,6.5])
    fig2.savefig('fig/estimation_attack%d.png' % attackerNum)
    plt.show()
