# Assaf Jeremiah 328458054
import numpy as np
import random

# generates a matrix containing the numbers 1 to num squared in a random order.
def generate_matrix(n):
    mat = np.arange(1, n * n + 1, 1)
    np.random.shuffle(mat)
    mat = mat.reshape(n, n)
    return mat

# generate a list of p_size matrixes of size s using the generate_matrix method.
def generate_population(p_size, s):
    p = [generate_matrix(s) for _ in range(p_size)]
    return p

# calculate the fitness of a matrix of size s for the magic square problem.
#
# formula - |sum - magic_constant|
# sum is for the rows columns and main diagonals.
#
# the lower the better, 0 means we got the solution.
def fitness(matrix, s):
    magic_constant = s * (s * s + 1) // 2
    fit = 0

    # rows and columns
    for i in range(s):
        fit += abs(np.sum(matrix[i, :]) - magic_constant)  # row
        fit += abs(np.sum(matrix[:, i]) - magic_constant)  # column

    # diagonals
    fit += abs(np.sum(np.diag(matrix)) - magic_constant)
    fit += abs(np.sum(np.diag(np.fliplr(matrix))) - magic_constant)

    return fit


# calculate the fitness of a matrix of size s for the perfect magic square problem.
#
# formula - |sum - magic_constant| + |tbt_sum - tbt_constant| + |mirrored_num - mirror_constant|
# sum is for the rows columns and main diagonals.
# mirrored_num is for the pairs whose distance from the main diagonal is n/2.
# tbt_sum is for the two by two matrix.
#
# the lower the better, 0 means we got the solution.
def fitness_ps(matrix, s):
    magic_constant = s * (s * s + 1) // 2
    mirror_constant = s * s + 1
    tbt_constant = mirror_constant * 2
    fit = 0

    # rows and columns
    for i in range(s):
        fit += abs(np.sum(matrix[i, :]) - magic_constant)  # row
        fit += abs(np.sum(matrix[:, i]) - magic_constant)  # column

    # diagonals
    fit += abs(np.sum(np.diag(matrix)) - magic_constant)
    fit += abs(np.sum(np.diag(np.fliplr(matrix))) - magic_constant)

    # mirrored
    for i in range(s):
        if i + s // 2 >= s:
            continue
        fit += abs(matrix[i + s // 2, i] + matrix[i, i + s // 2] - mirror_constant)  # row

    # 2 by 2
    for i in range(s - 1):
        for j in range(s - 1):
            fit += abs(matrix[i, j] + matrix[i, j + 1] + matrix[i + 1, j] + matrix[i + 1, j + 1]  - tbt_constant)

    return fit

# calls the right fitness function based on m.
def fitness_helper(matrix, s, m):
    if m == 1:
        return fitness(matrix, s)
    else:
        return fitness_ps(matrix, s)

# get 2 parents and return the crossover between them in the following manner.
#
# the child will contain the first half of the first parent while the second part will be from the 2nd parent.
# in the case that the value of the 2nd parent is already in the 1st one, it will be replaced by the remaining,
# values by random.
#
# returns the result of the crossover as a matrix.
def crossover(m1, m2, s):
    sizes = s * s

    # first half from the first parent.
    fh = m1.flatten()[:sizes // 2]
    numbers = list(range(1, sizes + 1)) # track what values are left.

    # remove seen numbers.
    for n in fh:
        numbers.remove(n)

    # second half from the second parent.
    sh = m2.flatten()[sizes // 2:]

    # used to track indexes in sh that are duplicates.
    dup_index = []

    # go over sh, if is a duplicate append the index, otherwise mark sh[i] off numbers.
    for i in range(len(sh)):
        if sh[i] not in numbers:
            dup_index.append(i)
            continue
        numbers.remove(sh[i])

    # replace the duplicates by random selecting from the remaining options.
    for n in numbers:
        i = random.randint(0, len(dup_index) - 1)
        sh[dup_index[i]] = n
        del dup_index[i]

    # build the child matrix.
    child = np.concatenate([fh,sh]).reshape(s, s)

    return child

# gets a matrix and its size, mutate it s times.
#
# each mutation is done by choosing 2 random indexes and switching them.
#
# returns the matrix after s mutations.
def mutation(m1, s):
    new = m1.flatten()

    for _ in range(size):
        i = random.randint(0, len(new) - 1)
        j = random.randint(0, len(new) - 1)

        temp = new[i]
        new[i] = new[j]
        new[j] = temp

    new = new.reshape(s, s)

    return new

# gets a matrix and its size, try to optimize it s times.
#
# each optimization is done by choosing 2 random indexes and switching them if it improves fitness.
#
# returns the matrix after s optimizations.
def optimize(m1, s, p_m):
    new = m1.flatten()
    optimized = m1.flatten()

    for _ in range(size):
        i = random.randint(0, len(new) - 1)
        j = random.randint(0, len(new) - 1)


        temp = new[i]
        new[i] = new[j]
        new[j] = temp

        # keep the mutation if it's beneficial to the fitness.
        if fitness_helper(new.reshape(s, s), s, p_m) < fitness_helper(optimized.reshape(s, s), s, p_m):
            optimized = new.copy()
        else:
            new = optimized.copy()

    optimized = optimized.reshape(s, s)

    return optimized

# gets the current gen p size of the matrix s and random chance of mutation r.
#
# iterates over the population calculating its fitness and weights,
# steps over to the next generations via carrying over the best matrix,
# biased selection for breeding and random chance of mutation.
#
# returns the next population.
def handle_generation(p, s, r, p_m):
    new_gen = []
    weights = []
    total_fitness = 0
    b_f = fitness(p[0], s)
    b_m = p[0]

    # calculate fitness for every matrix.
    for m in p:
        fit = fitness_helper(m,s, p_m)
        total_fitness += fit
        weights.append(fit)
        if fit < b_f:
            b_f = fit
            b_m = m.copy()

    a_f = total_fitness / 100
    # adjust weights.
    for index in range(len(weights)):
        weights[index] = total_fitness - weights[index]

    new_gen.append(b_m)
    for _ in range(len(p) - 1):
        parents = random.choices(population=p, weights=weights, k=2)
        child = crossover(parents[0], parents[1], s)

        mut = random.randint(1, r)
        if mut == 1:
            child = mutation(child, s)

        new_gen.append(child)

    return new_gen, b_f, b_m, a_f

# gets the current gen p size of the matrix s and random chance of mutation r.
#
# iterates over the population calculating its optimized fitness and weights but doesnt pass it on,
# steps over to the next generations via carrying over the best matrix,
# biased selection for breeding and random chance of mutation.
#
# returns the next population.
def handle_generation_darwin(p, s, r, p_m):
    new_gen = []
    weights = []
    total_fitness = 0
    b_f = fitness(p[0], s)
    b_m = p[0]

    # calculate fitness for every matrix.
    for m in p:
        fit = fitness_helper(optimize(m, s, p_m),s, p_m)
        total_fitness += fit
        weights.append(fit)
        if fit < b_f:
            b_f = fit
            b_m = m.copy()

    a_f = total_fitness / 100
    # adjust weights.
    for index in range(len(weights)):
        weights[index] = total_fitness - weights[index]

    new_gen.append(b_m)
    for _ in range(len(p) - 1):
        parents = random.choices(population=p, weights=weights, k=2)
        child = crossover(parents[0], parents[1], s)

        mut = random.randint(1, r)
        if mut == 1:
            child = mutation(child, s)

        new_gen.append(child)

    return new_gen, b_f, b_m, a_f

# gets the current gen p size of the matrix s and random chance of mutation r.
#
# iterates over the population optimizes it and then calculating its fitness and weights,
# steps over to the next generations via carrying over the best matrix,
# biased selection for breeding and random chance of mutation.
#
# returns the next population.
def handle_generation_lamrak(p, s, r, p_m):
    new_gen = []
    weights = []
    total_fitness = 0
    b_f = fitness(p[0], s)
    b_m = p[0]

    # calculate fitness for every matrix.
    for index in range(len(p)):
        p[index] = optimize(p[index], s, p_m)
        fit = fitness_helper(p[index],s, p_m)
        total_fitness += fit
        weights.append(fit)
        if fit < b_f:
            b_f = fit
            b_m = p[index].copy()

    a_f = total_fitness / 100

    # adjust weights.
    for index in range(len(weights)):
        weights[index] = total_fitness - weights[index]

    new_gen.append(b_m)
    for _ in range(len(p) - 1):
        parents = random.choices(population=p, weights=weights, k=2)
        child = crossover(parents[0], parents[1], s)

        mut = random.randint(1, r)
        if mut == 1:
            child = mutation(child, s)

        new_gen.append(child)

    return new_gen, b_f, b_m, a_f

# config.
max_gen = 500
population_size = 100
inverted_mutation_rate = 4

size = int(input("Enter matrix size: "))
algo_mode = int(input("Enter gia option: \n 1. regular\n 2. darwin\n 3. lemrak\n"))
problem_mode = int(input("Enter the problem: \n 1. magic square\n 2. most perfect magic square\n"))

# first gen.
population = generate_population(population_size, size)

for i in range(max_gen):
    if algo_mode == 1:
        population, best_fitness, best_matrix, average_fitness = handle_generation(population, size, inverted_mutation_rate, problem_mode)
    elif algo_mode == 2:
        population, best_fitness, best_matrix, average_fitness = handle_generation_darwin(population, size, inverted_mutation_rate, problem_mode)
    else:
        population, best_fitness, best_matrix, average_fitness = handle_generation_lamrak(population, size, inverted_mutation_rate, problem_mode)
    print(f"iteration {i} best fitness { best_fitness} average fitness {average_fitness}")

    if best_fitness > 0:
        continue

    print(f"done in {i} gens, the magic square is: \n{best_matrix}")
    break

best_matrix = optimize(best_matrix, size, problem_mode)
best_fitness = fitness_helper(best_matrix, size, problem_mode)

print(f"best matrix with fitness of {best_fitness}:\n {best_matrix}")