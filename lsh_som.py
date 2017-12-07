import numpy as np
from lshash import LSHash
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import matplotlib.pyplot as plt


class LshSom :
    '''Planar grid with locality-sensitive hashing method'''
    def __init__(self, xdim, ydim, raw_input, neighborhood="Gaussian",sigma=0.2, nSize=0.5, nSizefinal=0.05):
        self.x = xdim
        self.y = ydim
        self.neighborhood = neighborhood
        self.ninput = 0
        self.nSize = nSize
        self.nSizefinal = nSizefinal
        self.sigma = sigma
        # random initialization
        self.grid = np.tile(np.min(raw_input, axis=0), (self.x * self.y, 1)) + \
                    (np.tile(np.max(raw_input, axis=0), (self.x * self.y, 1)) -
                     np.tile(np.min(raw_input, axis=0), (self.x * self.y, 1))) * \
                    (np.random.rand(self.x * self.y, raw_input.shape[1]))
        self.map = np.reshape(np.transpose(np.mgrid[0:1:np.complex(0, self.x), 0:1:np.complex(0, self.y)]),
                              (self.x * self.y, 2))
        self.umatrix = np.array([[]])
        self.init_lsh()

    def bmu_lsh(self, input):
        return self.lsh.query(input,1)[0][0][1]-1

    def init_lsh(self):
        self.lsh=LSHash(8, 9)
        for i, neuron in enumerate(self.grid):
            self.lsh.index(neuron, extra_data=i+1)

    def find_all_bmu(self,subset): # lsh bmu research with gaussian neighborhood function
        futur_grid = np.zeros((self.x * self.y, np.shape(self.grid)[1]))
        futur_weight = np.zeros((self.y * self.x, 1))
        for inp in subset:
            best=self.bmu_lsh(inp)
            mapdist = cdist(np.array([self.map[best, :]]), self.map, 'cityblock')
            gaus_n = np.round(np.exp(- (mapdist ** 2 / (2 * self.sigma ** 2))), 3)
            futur_grid += np.transpose(gaus_n) * inp
            futur_weight += np.transpose(gaus_n)
        return [futur_grid,futur_weight]

    def train_som(self,inputs, nIterations, mode):
        sigma_init = self.sigma
        nsizeinit = self.nSize
        self.ninput=np.shape(inputs)[0]
        for iterations in range(nIterations):
            futur_grid = np.zeros((self.x * self.y, np.shape(self.grid)[1]))
            futur_weight = np.zeros((self.y * self.x, 1))
            nexact=0
            for i in range(self.ninput):
                # Find the bmu
                if mode == "lsh":
                    try:
                        best=self.bmu_lsh(inputs[i, :])
                    except IndexError:
                        nexact+=1
                        best=np.argmin(np.sum(abs(np.tile(inputs[i, :], (len(self.grid), 1)) - self.grid), axis=1))
                elif mode=="exact":
                    best = np.argmin(np.sum(abs(np.tile(inputs[i, :], (len(self.grid), 1)) - self.grid), axis=1))
                else:
                    raise Exception("Unsupported mode : '%s', choose between lsh or exact" % mode)
                # Calculate distances between the best and the others neurons.
                mapdist = cdist(np.array([self.map[best, :]]), self.map, 'cityblock')
                if self.neighborhood == "Bubble":
                    neighbours = np.where(mapdist <= self.nSize, 1, 0)  # bubble
                    futur_grid += neighbours * inputs[i, :]
                    futur_weight += neighbours
                elif self.neighborhood == "Gaussian":
                    gaus_n = np.exp(- (mapdist ** 2 / (2 * self.sigma ** 2)))
                    futur_grid += np.transpose(gaus_n) * inputs[i, :]
                    futur_weight += np.transpose(gaus_n)
                else:
                    raise Exception("Unsupported neighborhood function '%s'" % self.neighborhood)
            # Update the weights
            self.grid = futur_grid / futur_weight
            # Modify neighborhood function rates gaussian case
            self.sigma = sigma_init * (1.0 - float(iterations) / nIterations)
            # Modify neighbourhood size buble case
            self.nSize = nsizeinit * np.power(self.nSizefinal / nsizeinit, float(iterations) / nIterations)
            # re-initialize the lsh with the new weights
            if mode != "exact":
                self.init_lsh()
                print "needs exact method for {} inputs at iteration {}".format(nexact, iterations)

    def train_test(self,inputs, nIterations=10):
        sigma_init = self.sigma
        nsizeinit = self.nSize
        error = {'num': [], 'dist_lsh': [], 'dist_ex': [],'iteration':[]}
        self.ninput = np.shape(inputs)[0]
        for iterations in range(nIterations):
            err = 0
            derr_lsh = 0
            derr_ex = 0
            nem=0
            futur_grid = np.zeros((self.x * self.y, np.shape(self.grid)[1]))
            futur_weight = np.zeros((self.y * self.x, 1))
            for i in range(self.ninput):
                try:
                    best = self.bmu_lsh(inputs[i, :])
                except IndexError:
                    nem+=1
                    best = np.argmin(np.sum(abs(np.tile(inputs[i, :], (len(self.grid), 1)) - self.grid), axis=1))
                beste = np.argmin(np.sum(abs(np.tile(inputs[i, :], (len(self.grid), 1)) - self.grid), axis=1))
                derr_lsh += np.sum(abs(inputs[i] - self.grid[best]))
                derr_ex += np.sum(abs(inputs[i] - self.grid[beste]))
                if best != beste:
                    err += 1
                mapdist = cdist(np.array([self.map[best, :]]), self.map, 'cityblock')
                gaus_n = np.exp(- (mapdist ** 2 / (2 * self.sigma ** 2)))
                futur_grid += np.transpose(gaus_n) * inputs[i, :]
                futur_weight += np.transpose(gaus_n)

            error['num'].append(err)
            error['dist_lsh'].append(float(derr_lsh)/self.ninput)
            error['dist_ex'].append(float(derr_ex)/self.ninput)
            error['iteration'].append(iterations + 1)
            print " Iteration {} ,nb exact method {}, nb error {}".format(iterations,nem, err)
            # Update the weights
            self.grid = futur_grid / futur_weight
            # Modify neighborhood function rates gaussian case
            self.sigma = sigma_init * (1.0 - float(iterations) / nIterations)
            # Modify neighbourhood size buble case
            self.nSize = nsizeinit * np.power(self.nSizefinal / nsizeinit, float(iterations) / nIterations)
            # re-initialize the lsh with the new weights
            self.init_lsh()
        self.plot_qe(error)

'''
    def multi_train(self,inputs, nIterations=10,processnumb=4):
        sigma_init = self.sigma
        for iterations in range(nIterations):
            subset=self.create_subset(inputs,processnumb)
            pool = Pool()
            results = pool.map(self.find_all_bmu, subset)
            futur_grid,futur_weight=self.merge(results)
            self.grid = futur_grid / futur_weight
            # Modify neighborhood function rates gaussian case
            self.sigma = sigma_init * (1.0 - float(iterations) / nIterations)
            # re-initialize the lsh with the new weights
            self.init_lsh()
'''
    @staticmethod
    def create_subset(dataset,procesnumb):
        args=[]
        bit_size=np.shape(dataset)[0]/procesnumb
        for i in range(procesnumb):
            if i != procesnumb - 1:
                args.append(dataset[bit_size * i:bit_size * (i + 1), :])
            else:
                rest = np.shape(dataset)[0] % procesnumb
                args.append(dataset[bit_size * i:bit_size * (i + 1) + rest, :])
        return args

    @staticmethod
    def import_matrix_on_txt_file(file_name):
        with open(file_name, 'r') as f_data_facs:
            next(f_data_facs)
            dt_facs = []
            for i in f_data_facs:
                list_i = i.split()
                i_float = [float(y) for y in list_i]
                dt_facs.append(i_float)
        arr_facs = np.asarray(dt_facs)
        return arr_facs

    @staticmethod
    def merge(pool_res):
        total_grid = np.zeros((self.x * self.y, np.shape(self.grid)[1]))
        total_weight = np.zeros((self.y * self.x, 1))
        for tup in pool_res:
            total_grid+=tup[0]
            total_weight+=tup[1]
        return total_grid,total_weight

    @staticmethod
    def plot_qe(dict_error):
        plt.plot(dict_error['iteration'], dict_error['dist_lsh'], 'g-', dict_error['iteration'], dict_error['dist_ex'], 'b-')
        plt.ylabel('Quantization Error')
        plt.xlabel('iteration')
        plt.show()