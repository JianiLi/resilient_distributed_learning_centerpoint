
from matplotlib import rc
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist, euclidean

from centerpoint.Centerpoint import *

rc('text', usetex=True)


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])


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


def points_plot(ax, theta, Xte, yte, mesh=True, colorscale=cmap_light,
                cdiscrete=cmap_bold, alpha=0.1, psize=10, zfunc=False, predicted=False):
    h = .02
    # X = np.concatenate((Xtr, Xte))
    X = Xte
    # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # plt.figure(figsize=(10,6))
    data = np.c_[xx.ravel(), yy.ravel()]

    Z = np.array([1 if Hypothesis(theta, i) >= 0.5 else 0 for i in data])
    ZZ = Z.reshape(xx.shape)
    if mesh:
        plt.pcolormesh(xx, yy, ZZ, cmap=cmap_light, alpha=alpha, axes=ax)
    if predicted:
        # showtr = Hypothesis(theta, Xtr)
        showte = Hypothesis(theta, Xte)
    else:
        # showtr = ytr
        showte = yte
    # ax.scatter(Xtr[:, 0], Xtr[:, 1], c=showtr-1, cmap=cmap_bold,
    #            s=psize, alpha=alpha,edgecolor="k")
    # and testing points
    ax.scatter(Xte[:, 0], Xte[:, 1], c=showte - 1, cmap=cmap_bold,
               alpha=alpha, marker="s", s=psize + 10)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    return ax, xx, yy


def get_raw_data_old(noise, outlier_rate=0.2, raw_attribute=2):
    attribute = raw_attribute
    num = 100

    noise0 = noise[0]
    noise1 = noise[1]
    # class 0
    data_0 = np.zeros((num, attribute + 1))
    mean = [2] * attribute
    # cov = np.ones((attribute, attribute))
    cov = noise0 * np.ones((attribute, attribute))
    x_0 = np.random.multivariate_normal(mean, cov, num)
    y_0 = np.zeros((num,)).astype(int)
    data_0[:, :-1] = x_0
    data_0[:, -1] = y_0
    for i in range(int(num / 2 * outlier_rate)):
        data_0[i, -1] = 1

    # class 1
    data_1 = np.zeros((num, attribute + 1))
    mean = [-2] * attribute
    # cov = np.ones((attribute, attribute))
    cov = noise1 * np.ones((attribute, attribute))
    x_1 = np.random.multivariate_normal(mean, cov, num)
    y_1 = np.ones((num,)).astype(int)
    data_1[:, :-1] = x_1
    data_1[:, -1] = y_1
    for i in range(int(num / 2 * outlier_rate)):
        data_1[i, -1] = 0

    data = np.zeros((2 * num, attribute + 1))
    data[0:num] = data_0
    data[num:] = data_1

    np.random.shuffle(data)

    x = data[:, 0:attribute]
    y = data[:, attribute]

    # train_data = data[0: int(0.7*num),:]
    # test_data = data[int(0.7 * num):,:]

    return x, y


def get_raw_data(num, noise, outlier_rate=0.2, raw_attribute=2):
    attribute = raw_attribute

    noise0 = noise[0]
    noise1 = noise[1]
    # class 0
    data_0 = np.zeros((num, attribute + 1))
    mean = [1] * attribute
    # cov = np.ones((attribute, attribute))
    cov = noise0 * np.ones((attribute, attribute))
    x_0 = np.random.multivariate_normal(mean, cov, num)
    y_0 = np.zeros((num,)).astype(int)
    data_0[:, :-1] = x_0
    data_0[:, -1] = y_0
    for i in range(num):
        if np.random.random() < outlier_rate:
            data_0[i, -1] = 1

    # class 1
    data_1 = np.zeros((num, attribute + 1))
    mean = [-1] * attribute
    # cov = np.ones((attribute, attribute))
    cov = noise1 * np.ones((attribute, attribute))
    x_1 = np.random.multivariate_normal(mean, cov, num)
    y_1 = np.ones((num,)).astype(int)
    data_1[:, :-1] = x_1
    data_1[:, -1] = y_1
    # for i in range(int(num/2*outlier_rate)):
    #     #     data_1[i, -1] = 0
    for i in range(num):
        if np.random.random() < outlier_rate:
            data_1[i, -1] = 0

    data = np.zeros((2 * num, attribute + 1))
    data[0:num] = data_0
    data[num:] = data_1

    np.random.shuffle(data)

    x = data[:, 0:attribute]
    y = data[:, attribute]

    # train_data = data[0: int(0.7*num),:]
    # test_data = data[int(0.7 * num):,:]

    return x, y


##The sigmoid function adjusts the cost function hypotheses to adjust the algorithm proportionally for worse estimations
def Sigmoid(z):
    G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0 * z))))
    return G_of_Z


##The hypothesis is the linear combination of all the known factors x[i] and their current estimated coefficients theta[i]
##This hypothesis will be used to calculate each instance of the Cost Function
def Hypothesis(theta, x):
    z = 0
    for i in range(len(theta)):
        z += x[i] * theta[i]
    return Sigmoid(z)


##For each member of the dataset, the result (Y) determines which variation of the cost function is used
##The Y = 0 cost function punishes high probability estimations, and the Y = 1 it punishes low scores
##The "punishment" makes the change in the gradient of ThetaCurrent - Average(CostFunction(Dataset)) greater
def Cost_Function(X, Y, theta, m):
    sumOfErrors = 0
    for i in range(m):
        xi = X[i]
        hi = Hypothesis(theta, xi)
        if Y[i] == 1:
            error = Y[i] * math.log(hi)
        elif Y[i] == 0:
            error = (1 - Y[i]) * math.log(1 - hi)
        sumOfErrors += error
    const = -1 / m
    J = const * sumOfErrors
    # print('cost is ', J)
    return J


##This function creates the gradient component for each Theta value
##The gradient is the partial derivative by Theta of the current value of theta minus
##a "learning speed factor aplha" times the average of all the cost functions for that theta
##For each Theta there is a cost function calculated for each member of the dataset
def Cost_Function_Derivative(X, Y, theta, j, m, alpha):
    sumErrors = 0
    for i in range(m):
        xi = X[i]
        xij = xi[j]
        hi = Hypothesis(theta, X[i])
        error = (hi - Y[i]) * xij
        sumErrors += error
    m = len(Y)
    constant = float(alpha) / float(m)
    J = constant * sumErrors
    return J


##For each theta, the partial differential
##The gradient, or vector from the current point in Theta-space (each theta value is its own dimension) to the more accurate point,
##is the vector with each dimensional component being the partial differential for each theta value
def Gradient_Descent(X, Y, theta, m, alpha):
    new_theta = []
    constant = alpha / m
    for j in range(len(theta)):
        CFDerivative = Cost_Function_Derivative(X, Y, theta, j, m, alpha)
        new_theta_value = theta[j] - CFDerivative
        new_theta.append(new_theta_value)
    return new_theta


##The high level function for the LR algorithm which, for a number of steps (num_iters) finds gradients which take
##the Theta values (coefficients of known factors) from an estimation closer (new_theta) to their "optimum estimation" which is the
##set of values best representing the system in a linear combination model
def Logistic_Regression(X, Y, alpha, theta, num_iters):
    m = len(Y)
    for x in range(num_iters):
        new_theta = Gradient_Descent(X, Y, theta, m, alpha)
        theta = new_theta
        # if x % 100 == 0:
        # here the cost function is used to present the final hypothesis of the model in the same form for each gradient-step iteration
        # loss = Cost_Function(X, Y, theta, m)
        # print('theta ', theta)

    return theta


# These are the initial guesses for theta as well as the learning rate of the algorithm
# A learning rate too low will not close in on the most accurate values within a reasonable number of iterations
# An alpha too high might overshoot the accurate values or cause irratic guesses
# Each iteration increases model accuracy but with diminishing returns,
# and takes a signficicant coefficient times O(n)*|Theta|, n = dataset length


if __name__ == '__main__':

    np.random.seed(1)
    random.seed(1)
    N = 10
    iteration = 501
    raw_attribute = 2
    pca_attribute = 2
    attacker_num = 3 # 1
    attackers = random.choices(range(N), k=attacker_num)
    normalAgents = [k for k in range(N) if k not in attackers]
    noise0 = np.random.rand(raw_attribute, raw_attribute)
    noise1 = np.random.rand(raw_attribute, raw_attribute)

    # RANDOM OUTLIER RATE IN [0, 0.3]
    # Outlier_rate = np.random.uniform(low=0.0, high=0.3, size=(10,))
    # UNIFORM OUTLIER RATE 0.1
    Outlier_rate = 0.2 * np.ones((N, ))

    X_test, Y_test = get_raw_data(500, [noise0, noise1], outlier_rate=0, raw_attribute=2)

    # mu_vd = 0
    # sigma_vd2 = 0.5 + 0.5 * np.random.random((N, 1))
    # sigma_vd2[random.sample(range(N), N // 2)] = 5

    Loss_all_rules = {}
    rules = ['Non-coop SGD', 'Centerpoint', 'CM', 'Average', 'GM']
    # rules = ['CM','Average','Non-coop SGD']

    for rule in rules:
        np.random.seed(1)
        random.seed(1)

        Theta = {}
        initial_theta = [0, 0]

        for k in range(N):
            theta = initial_theta
            Theta[k] = theta

        Theta_inter = Theta
        Loss = np.zeros((iteration, N))

        for i in range(iteration):
            print("iteration ", i)
            for k in range(N):
                if k in attackers:
                    Theta[k] = [1, -1]
                    continue
                X_train, Y_train = get_raw_data(1, [noise0, noise1], outlier_rate=Outlier_rate[k], raw_attribute=2)
                # X_train = X_train + np.random.normal(mu_vd, sigma_vd2[k], 1)

                # creating testing and training set
                # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

                alpha = 0.01
                iterations = 1
                Theta[k] = Logistic_Regression(X_train, Y_train, alpha, Theta[k], iterations)

                train_loss = Cost_Function(X_train, Y_train, Theta[k], len(Y_train))
                loss = Cost_Function(X_test, Y_test, Theta[k], len(Y_test))
                print('Agent %d, Train loss is %.3f, Test loss is %.3f' % (k, train_loss, loss))
                Loss[i, k] = loss

                # with plt.style.context(['science']):
                #     plt.figure(figsize=(4,4))
                #     ax = plt.gca()
                #     pos = np.where(Y_test == 1)
                #     neg = np.where(Y_test == 0)
                #     ax.scatter(X_test[pos, 0], X_test[pos, 1], marker='o', c='b', label="Label 1")
                #     ax.scatter(X_test[neg, 0], X_test[neg, 1], marker='x', c='r', label="Label 0")
                #     plt.xlabel('Feature 1', size=18)
                #     plt.ylabel('Feature 2', size=18)
                #     plt.legend(loc='upper left')
                #     plt.tight_layout()
                #     # points_plot(ax, Theta[k], X_train, X_test, Y_train, Y_test, alpha=0.2)
                #     plt.xlim([-3, 3])
                #     plt.ylim([-3, 3])
                #     plt.show()

            print("average test loss:", np.mean(Loss[i, [k for k in normalAgents]]))
            if rule == "Non-coop SGD":
                for k in range(N):
                    Theta_inter[k] = Theta[k]
            elif rule == "Average":
                theta_inter = np.mean([x for x in Theta.values()], axis=0)
                for k in range(N):
                    Theta_inter[k] = theta_inter
            elif rule == "CM":
                theta_inter = np.median([x for x in Theta.values()], axis=0)
                for k in range(N):
                    Theta_inter[k] = theta_inter
            elif rule == "GM":
                theta_inter = geometric_median(np.array([Theta[k] for k in range(N)]))
                for k in range(N):
                    Theta_inter[k] = theta_inter
            elif rule == "Centerpoint":
                centerPoint = Centerpoint()
                theta_inter = centerPoint.getSafeCenterPoint([Point(Theta[k][0], Theta[k][1]) for k in range(N)])
                for k in range(N):
                    Theta_inter[k][0] = theta_inter.x
                    Theta_inter[k][1] = theta_inter.y

            for k in range(N):
                Theta = Theta_inter

        # Loss_all_rules[rule] = np.mean(Loss[:, [k for k in normalAgents]], 1)
        Loss_all_rules[rule] = Loss[:, [k for k in normalAgents]]

        # with plt.style.context(['science']):
        #     plt.figure(figsize=(4, 4))
        #     ax = plt.gca()
        #     pos = np.where(Y_test == 1)
        #     neg = np.where(Y_test == 0)
        #     ax.scatter(X_test[pos, 0], X_test[pos, 1], marker='o', c='b', label="Label 1")
        #     ax.scatter(X_test[neg, 0], X_test[neg, 1], marker='x', c='r', label="Label 0")
        #     plt.xlabel('Feature 1', size=18)
        #     plt.ylabel('Feature 2', size=18)
        #     plt.legend(loc='upper left')
        #     plt.tight_layout()
        #     points_plot(ax, Theta[0],  X_test,  Y_test, alpha=0.2)
        #     plt.xlim([-3, 3])
        #     plt.ylim([-3, 3])
        #     plt.show()

    with plt.style.context(['science']):
        plt.figure(figsize=(4, 3.8))
        # ax.set_aspect('equal', adjustable='box')
        plt.ylabel("Test Loss", size=18)
        plt.xlabel("iteration $i$", size=18)
        plt.xlim([0, iteration - 1])
        if attacker_num != 0:
            for r in rules:
                plt.plot(np.mean(Loss_all_rules[r], 1), linewidth=2, label=r + ", Mean")
                if r == "Non-coop SGD":
                    plt.fill_between(range(iteration), np.max(Loss_all_rules[r], 1), np.min(Loss_all_rules[r], 1),
                                     facecolor='blue', alpha=0.1, label="Non-coop SGD, Range")
        else:
            for r in rules:
                if r in ["Non-coop SGD", "Centerpoint"]:
                    plt.plot(np.mean(Loss_all_rules[r], 1), linewidth=2, label=r+", Mean")
                else:
                    plt.plot(np.mean(Loss_all_rules[r], 1), linewidth=2, linestyle='--', label=r+", Mean")
                if r == "Non-coop SGD":
                    plt.fill_between(range(iteration), np.max(Loss_all_rules[r], 1), np.min(Loss_all_rules[r], 1),
                                     facecolor='blue', alpha=0.1, label="Non-coop SGD, Range")

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
