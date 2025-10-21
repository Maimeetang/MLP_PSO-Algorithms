import copy
import math
import random
from util import zeros_like, minus_pairs, add_pairs, scale

random.seed(1)

def random_weight():
    val = 0.0
    while val == 0.0:
        val = random.uniform(-1, 1)
    return val

def sigmoid(x): return 1 / (1 + math.exp(-x))

def tanh(x): return math.tanh(x)

class mlp_pso:
    
    def __init__(self, input_size, hidden_layers_list, output_size, swarm_size):

        self.input_size = input_size
        self.output_size = output_size
        self.swarm_size = swarm_size

        self.layers = [input_size] + hidden_layers_list + [output_size]

        # pairs of layer node: [2, 3, 4, 1] => [(2, 3), (3, 4), (4, 1)]
        self.weight_shapes = []
        for i in range(len(self.layers) - 1):
            self.weight_shapes.append((self.layers[i], self.layers[i+1]))

        self.num_weights = 0
        for (node_in, node_out) in self.weight_shapes:
            self.num_weights += node_in * node_out

        self.num_bias = 0
        for (node_in, node_out) in self.weight_shapes:
            self.num_bias += node_out

        self.swarm = []
        for _ in range(swarm_size):
            particle = [random_weight() for _ in range(self.num_weights)]
            particle = self.weight_vector(particle)

            bias = []
            for (node_in, node_out) in self.weight_shapes:
                bias.append([random_weight() for _ in range(node_out)])

            self.swarm.append((particle,bias))
            
        self.particle_size = self.num_weights + self.num_bias

    # function: reshape 1D particle into layer-wise weight matrices
    # weight_vector :: [particle] => [layer][input_node][output_node] particle
    def weight_vector(self, particle):
        matrices = []
        pointer = 0
        for (node_in, node_out) in self.weight_shapes:
            matrix = []
            for _ in range(node_in):
                row = []
                for _ in range(node_out):
                    row.append(particle[pointer])
                    pointer += 1
                matrix.append(row)
            matrices.append(matrix)
        return matrices
    
    # feed_forward ::  [x] -> [particle] => [y]
    def feed_forward(self, input, particle, bias):
        output = input

        layer = 0
        for (node_in, node_out) in self.weight_shapes:
            new_output = []
            for o in range(node_out):
                node_sum = 0
                for i in range(node_in):
                    node_sum += particle[layer][i][o] * output[i]
                node_sum += bias[layer][o]
                node_sum = tanh(node_sum)
                new_output.append(node_sum)
            output = new_output
            layer += 1

        return output
    
    def fitness_func(self, input, target, particle, bias):
        output = self.feed_forward(input, particle, bias)
        mae = (1/len(output)) * sum(abs(t-o) for o,t in zip(output, target))
        return mae
    
    def l_best_algorithm(self, sample_list, c1, c2, inertia_weight, t_max):
        
        t = 0

        p_best = [float('inf') for _ in range(self.swarm_size)]
        x_p_best = [None]*len(self.swarm)

        l_best = [float('inf') for _ in range(self.swarm_size)]
        x_l_best = [None]*len(self.swarm)

        list_v = zeros_like(self.swarm)

        while t < t_max:

            # Evaluate the performance F of each particle
            swarm_fitness = []
            for (particle, bias) in self.swarm:
                sum_mae = 0
                for sample in sample_list:
                    input, target = sample
                    sum_mae += self.fitness_func(input,target,particle,bias)
                swarm_fitness.append(sum_mae/len(sample_list))

            # Compare the performance of each individual to its best performance
            for i in range(len(swarm_fitness)):

                # update p_best
                if swarm_fitness[i] < p_best[i]:
                    p_best[i] = swarm_fitness[i]
                    x_p_best[i] = copy.deepcopy(self.swarm[i])
                
                # update l_best left Ring topology
                if i == 0:
                    index = self.swarm_size - 1
                    if swarm_fitness[i] < l_best[index]:
                        l_best[index] = swarm_fitness[i]
                        x_l_best[index] = copy.deepcopy(self.swarm[i])
                else:
                    if swarm_fitness[i] < l_best[i-1]:
                        l_best[i-1] = swarm_fitness[i]
                        x_l_best[i-1] = copy.deepcopy(self.swarm[i])
                
                # update l_best center Ring topology
                if(swarm_fitness[i] < l_best[i]):
                    l_best[i] = swarm_fitness[i]
                    x_l_best[i] = copy.deepcopy(self.swarm[i])

                # update l_best right Ring topology
                if i == self.swarm_size - 1:
                    if swarm_fitness[i] < l_best[0]:
                        l_best[0] = swarm_fitness[i]
                        x_l_best[0] = copy.deepcopy(self.swarm[i])
                else:
                    if swarm_fitness[i] < l_best[i+1]:
                        l_best[i+1] = swarm_fitness[i]
                        x_l_best[i+1] = copy.deepcopy(self.swarm[i])

            # Change the velocity vector of each particle
            list_v_new = []
            swarm_new = []
            for i in range(self.swarm_size):
                r1 = random.uniform(0,1)
                r2 = random.uniform(0,1)
                
                # find ρ(x_p_best[i] - x[i])
                x1 = minus_pairs(x_p_best[i],self.swarm[i])
                x1 = scale(x1, c1*r1)

                # find ρ(x_l_best[i] - x[i])
                x2 = minus_pairs(x_l_best[i],self.swarm[i])
                x2 = scale(x2, c2*r2)

                # find new velocity
                if isinstance(list_v, list):
                    v_new = scale(list_v[i], inertia_weight)
                    v_new = add_pairs(v_new, add_pairs(x2,x1))
                    list_v_new.append(v_new)
                else:
                    raise TypeError("invalid list_v")
            
            #  Move each particle to a new position
            for i in range(self.swarm_size):
                particle_new = add_pairs(self.swarm[i],list_v_new[i])
                swarm_new.append(particle_new)

            list_v = list_v_new
            self.swarm = swarm_new

            if t+1 == 1:
                print("ep", t+1)
                
            if (t+1) % 10 == 0:
                print("ep", t+1)

            t += 1

        g_idx = min(range(self.swarm_size), key=lambda i: p_best[i])
        self.g_best_fitness = p_best[g_idx]
        self.g_best_particle = copy.deepcopy(x_p_best[g_idx])