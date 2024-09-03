"""
============================================================
Single-Machine Scheduling ILS 
============================================================
Author: Giovanni Cesar Meira Barboza
Date: August 2024
Description:
This script optimizes job scheduling with an Iterated Local Search algorithm. It supports 
evaluating swap and insertion moves, calculating completion times, and modifying job sequences.

Based on the "Iterated local search based on multi-type perturbation for single-machine earliness/tardiness scheduling" by Qin et al. 2015.
Tailored to tackle Biskup's 2001 "Benchmarks for scheduling on a single machine against restrictive and unrestrictive common due dates".

Scheduling Problem with Weighted Tardiness and Earliness

f.o.    min = a*E + b*T
s.a     Ti >= si + pi = d
        Ei >= d - si = pi
   si + pi <= sk + R*(1 - xik)
   sk + pk <= si + R*xik
Ti, Ei, si >= 0
        xik = 0 or 1

a = earliness weight
b = tardiness weight
E = earliness
T = tardiness
s = starting time
p = processing time
d = common due date
i, k = jobs
x = whether the job i preceds k
R = large constant

V-Shape property: the best solutions are ordered so that the ratios of the jobs pi/ai are in 
decreasing order in the early jobs set and in increasing pi/bi order in the late jobs set.
============================================================
"""

# ==================================================================
#                     CONFIGURATION
# ==================================================================

import random
import time

class Job:
    def __init__(self, processing_time, weight_e, weight_t):
        self.processing_time = processing_time                  # p
        self.weight_e = weight_e                                # a
        self.weight_t = weight_t                                # b
        self.ratio_e = self.processing_time / self.weight_e     # p/a
        self.ratio_t = self.processing_time / self.weight_t     # p/b

# Test values
# Define jobs from a three-column list: [processing_time, weight_e, weight_t]
job_data = [[15,5,1],[16,10,9],[20,2,13],[13,8,7],[6,10,10],[18,5,6],[11,2,14],[4,8,9],[16,5,3],[11,5,11],[10,7,2],[7,7,12],[11,6,11],[1,6,1],[14,8,9],[10,10,8],[5,8,14],[19,8,4],[9,8,11],[17,7,4]]
    # sch20, k = 5
    # Add more job data as needed

# Convert list to Job objects
jobs = [Job(processing_time, weight_e, weight_t) for processing_time, weight_e, weight_t in job_data]

# ==================================================================
#                     AUXILIARY FUNCTIONS
# ==================================================================      

# Total cost calculation

def total_cost(jobs, A, B):
    # Require: list of jobs, sequence of early jobs (A) with the last job ending at the common due date, sequence of late jobs (B) and common due date
    # Ensure: total cost of the sequence (A+B)

    cost_A = 0
    cost_B = 0

    A_rev = [*reversed(A)]

    current_time = 0
    for i in range(len(A)):
        cost_A += jobs[A_rev[i]].weight_e * current_time
        current_time += jobs[A_rev[i]].processing_time

    current_time = jobs[B[0]].processing_time
    for i in range(len(B)):
        cost_B += jobs[B[i]].weight_t * current_time
        if i < len(B) - 1: current_time += jobs[B[i + 1]].processing_time

    return cost_A + cost_B

# Insertion and Swap functions

def insert_job(A, B, j, i):
    # Insert element j just before element i into C (new A) or D (new B)
    C = A[:]
    D = B[:]
    if i < 0:   # Negative i is used to signal insertion on (after) the last element
        if j in C:
            j = C.pop(C.index(j))
            D.append(j)
        else:
            j = D.pop(D.index(j))
            C.append(j)
    else:
        if j in C:
            j = C.pop(C.index(j))
            D.insert(D.index(i), j)
        else:
            j = D.pop(D.index(j))
            C.insert(C.index(i), j)
    return C, D

def swap_job(A, B, j, i):
    # Swap the elements j and i into C (new A) and D (new B)
    C = A[:]
    D = B[:]
    if j in C:
        j_index = C.index(j)
        i_index = D.index(i)
        C[j_index], D[i_index] = D[i_index], C[j_index]
    else:
        j_index = D.index(j)
        i_index = C.index(i)
        D[j_index], C[i_index] = C[i_index], D[j_index]
    return C, D

# ==================================================================
#                        V-SHAPE CANDIDATES
# ==================================================================
def vshape_candidates(jobs, A, B, common_due_date, j, threshold, insert):
    # Require: jobs, sequence of early jobs (A) with the last job ending at the common due date, sequence of late jobs (B), common due date, index of the job j (in the jobs list)
    # Ensure: v-shape abiding candidates for local search so that the early jobs processing time sum does not exceet the common due date
        
    candidates = []
    rm_cand = []    # list of candidates to be removed due to a violation

    # Calculate current early jobs processing time
    A_lenght = 0
    for i in A:
        A_lenght += jobs[i].processing_time

    # Insertion evaluation
    found = False   # found candidate
    if insert:    
        if j in A:  # j is early
            for idx in range(len(B)):
                if found:
                    if jobs[B[idx]].ratio_t == jobs[j].ratio_t:
                        candidates.append(B[idx])   # multiple insertions (same ratio)
                    else:
                        break
                if jobs[j].ratio_t <= jobs[B[idx]].ratio_t and not found:
                    candidates.append(B[idx])   # regular insertion
                    found = True
                if idx == len(B) - 1 and not found:
                    candidates.append(-1)    # insertion at the end

        else:   # j is late
            if A_lenght + jobs[j].processing_time <= common_due_date:   # check if due date is exceeded by insertion of j
                for idx in range(len(A)-1, -1, -1):
                    if found:
                        if jobs[A[idx]].ratio_e == jobs[j].ratio_e:
                            candidates.append(A[idx])   # multiple insertions (same ratio)
                    if jobs[j].ratio_e <= jobs[A[idx]].ratio_e and not found:
                        found = True
                        if idx == len(A) - 1:
                            candidates.append(-1)   # insertion ends at due date
                        else:
                            candidates.append(A[idx + 1])   # regular insertion
                    if idx == 0 and not found:
                        candidates.append(A[0])    # insertion at the start
    
    # Swap evaluation
    else:
        if j in A:  # j is early
            j_index = A.index(j)
            if j_index == 0:
                prev_eval = 999
            else:
                prev_eval = jobs[A[j_index - 1]].ratio_e

            if j_index == len(A) - 1:
                next_eval = 0
            else:
                next_eval = jobs[A[j_index + 1]].ratio_e
            
            #print(f'j = {j}, next = {next_eval}, prev = {prev_eval}')
            # Check if i can be inserted into the ratio space of j and if it would not exceed the due date
            for i in B:
                if jobs[i].ratio_e <= prev_eval and jobs[i].ratio_e >= next_eval and A_lenght + jobs[i].processing_time - jobs[j].processing_time <= common_due_date:
                    candidates.append(i)

            # Check if j can be inserted into the ratio space of i
            for i in candidates:
                i_index = B.index(i)
                if i_index == 0:
                    prev_eval = 0
                else:
                    prev_eval = jobs[B[i_index - 1]].ratio_t

                if i_index == len(B) - 1:
                    next_eval = 999
                else:
                    next_eval = jobs[B[i_index + 1]].ratio_t
                
                if jobs[j].ratio_t < prev_eval or jobs[j].ratio_t > next_eval:
                    rm_cand.append(i)
                    
        else:   # j is late
            j_index = B.index(j)
            if j_index == 0:
                prev_eval = 0
            else:
                prev_eval = jobs[B[j_index - 1]].ratio_t

            if j_index == len(B) - 1:
                next_eval = 999
            else:
                next_eval = jobs[B[j_index + 1]].ratio_t
            
            # Check if i can be inserted into the ratio space of j and if the insertion of j in A would not exceed the due date
            for i in A:
                if jobs[i].ratio_t >= prev_eval and jobs[i].ratio_t <= next_eval and A_lenght + jobs[j].processing_time - jobs[i].processing_time <= common_due_date:
                    candidates.append(i)

            # Check if j can be inserted into the ratio space of i
            for i in candidates:
                i_index = A.index(i)
                if i_index == 0:
                    prev_eval = 999
                else:
                    prev_eval = jobs[A[i_index - 1]].ratio_e

                if i_index == len(A) - 1:
                    next_eval = 0
                else:
                    next_eval = jobs[A[i_index + 1]].ratio_e
                
                if jobs[j].ratio_e > prev_eval or jobs[j].ratio_e < next_eval:
                    rm_cand.append(i)

    candidates = [i for i in candidates if i not in rm_cand]
    rm_cand = []

    # Remove candidates that violate the threshold

    sequence = A + B
    for i in candidates:
        if i < 0:
            if j in A:
                if abs((len(sequence) - 1) - sequence.index(j)) > threshold:
                    rm_cand.append(i)
            else:
                if abs(0 - sequence.index(j)) > threshold:
                    rm_cand.append(i)
        else:
            if abs(sequence.index(i) - sequence.index(j)) > threshold:
                rm_cand.append(i)
    candidates = [i for i in candidates if i not in rm_cand]
    
    return candidates

# ==================================================================
#                      EVALUATION PROCEDURE
# ==================================================================

# Here there is room for improvement as the evaluation procedure proposed by Qin et al. 2015 does not apply to this problem; one could propose a better method than calculating the total cost for each candidate

def evaluation_procedure(jobs, A, B, candidates, j, insert, tabu):
    # Require: list of jobs, A, B, candidates for movement, index of the job j (in the sequence) to evaluate swaps and insertions possible within the sequence, insert (True -> insert; False -> swap), tabu insert (True -> tabu perturbation; False -> normal evaluation)
    # Ensure: minimum cost of objective function, operation used (swap or insertion), index of the job swapped with or inserted before
    
    if len(candidates) == 0: return []  # No candidates, empty evaluation

    F = []
    f = total_cost(jobs, A, B)

    for k in candidates:
        if insert:
            A_new, B_new = insert_job(A, B, j, k)
            f_new = total_cost(jobs, A_new, B_new)
            F.append([f_new, insert, k])     # (new objective function, s for swap, index to swap with)
        else:
            A_new, B_new = swap_job(A, B, j, k)
            f_new = total_cost(jobs, A_new, B_new)
            F.append([f_new, insert, k])     # (new objective function, s for swap, index to swap with)

    k = min(F, key=lambda x: x[0])

    if tabu:    # Allows tabu search to admit worse solution
        return k
    else:
        if k[0] < f: return k
        else: return []

# ==================================================================
#                           LOCAL SEARCH
# ==================================================================

# Local search best move
# Tries all the candidates, stores moves that improve the cost. Takes best improving move. Do it again until there are no improving moves left

def local_search(jobs, A, B, common_due_date, threshold_swaps, threshold_inserts, insert_probability):
    # Require: list of jobs, A, B, common due date, number of jobs for swaps and inserts neighborhood, probability of choosing insert 
    # Ensure: local optimum sequence following limited neighborhood local search 

    insert = random.random() < insert_probability   # randomly select between insertion or swap

    sequence = A + B

    if insert: threshold = threshold_inserts
    else: threshold = threshold_swaps
    
    while True:
        moves = []
        for j in sequence:
            candidates = vshape_candidates(jobs, A, B, common_due_date, j, threshold, insert)
            if len(candidates) > 0:
                if insert:  # Insertion move
                    move = evaluation_procedure(jobs, A, B, candidates, j, True, False)
                    if len(move) > 0:
                        move.append(j)
                        moves.append(move)              
                else:   # Swap move
                    move = evaluation_procedure(jobs, A, B, candidates, j, False, False)
                    if len(move) > 0:
                        move.append(j)
                        moves.append(move)  
                            
        # Perform move that minimizes cost degradation
        if len(moves) > 0:
            move = min(moves, key=lambda x: x[0])
            if insert:
                A, B = insert_job(A, B, move[3], move[2])
            else:
                A, B = swap_job(A, B, move[3], move[2])
        else:
            break   # if there are no moves left the local search is over
    
    return A, B

# ==================================================================
#                      TABU-BASED PERTURBATION
# ==================================================================

def tabu_perturbation(jobs, A, B, common_due_date, tabu_parameters, threshold_inserts, threshold_swaps, insert_probability, best_cost, L1):
    # Similar to local search but with tabu elements

    alpha_1, alpha_2 = tabu_parameters

    # Gamma: each time a job is moved from its current position i to position j, it is forbidden to be placed back to i for gamma iterations:
    gamma_inserts = int(alpha_1 * threshold_inserts + random.uniform(0, alpha_2 * threshold_inserts))
    gamma_swaps = int(alpha_1 * threshold_swaps + random.uniform(0, alpha_2 * threshold_swaps))

    tabu_list = []  # List of prohibited moves, only to be ignored if solution is better than current
    
    insert = random.random() < insert_probability   # randomly select between insertion or swap

    if insert: 
        threshold = threshold_inserts
        gamma = gamma_inserts
    else: 
        threshold = threshold_swaps
        gamma = gamma_swaps

    improved = False
    A_best = []
    B_best = []

    sequence = A + B

    # Make the moves that minimize cost degradation, as long as they are not on tabu-list
    
    for _ in range(L1):
        if len(tabu_list) == gamma + 1: tabu_list = tabu_list[1:]   # Remove first element of tabu list if maximum size/iterations gamma is reached
        moves = []
        for j in sequence:
            candidates = vshape_candidates(jobs, A, B, common_due_date, j, threshold, insert)
            if len(candidates) > 0:
                if insert:  # Insertion move
                    move = evaluation_procedure(jobs, A, B, candidates, j, True, True)
                    move.append(j)
                    if move[0] < best_cost:     # if improved, ignore tabu list, just do the move
                        improved = True
                        A_best = A[:]
                        B_best = B[:]
                        best_cost = total_cost(jobs, A_best, B_best)
                        if j in tabu_list:      # it's possible that the move is already in the tabu list, in which case we remove the element and put it in first
                            tabu_list.pop(tabu_list.index(j))
                        tabu_list.append(j)
                        moves.append(move)
                    else:
                        if  j not in tabu_list:
                            moves.append(move)
                    
                else:
                    move = evaluation_procedure(jobs, A, B, candidates, j, False, True)
                    move.append(j)
                    if move[0] < best_cost:     # if improved, ignore tabu list, just do the move
                        improved = True
                        A_best = A[:]
                        B_best = B[:]
                        best_cost = total_cost(jobs, A_best, B_best)
                        if j in tabu_list:      # it's possible that the move is already in the tabu list, in which case we remove the element and put it in first
                            tabu_list.pop(tabu_list.index(j))
                        tabu_list.append(j)
                        moves.append(move)
                    else:
                        if  j not in tabu_list:
                            moves.append(move)

        # Perform move that minimizes cost degradation, even if it  worsens the cost
        if len(moves) > 0:
            move = min(moves, key=lambda x: x[0])
            if insert:
                A, B = insert_job(A, B, move[3], move[2])
            else:
                A, B = swap_job(A, B, move[3], move[2])
            tabu_list.append(move[3])

    if improved:
        A, B = A_best, B_best

    return A, B, improved

# ==================================================================
#                   CONSTRUCTION-BASED PERTURBATION
# ==================================================================

def harmonic_select(list):
    n = len(list)
    
    # Compute Harmonic Weights
    weights = [1/k for k in range(1, n+1)]
    
    # Select an element based on the weights
    selected_element = random.choices(list, weights=weights, k=1)[0]
    
    return selected_element

def construction_perturbation(jobs, A, B, common_due_date):
    # List all possible insert moves
    # Order them by decreasing cost
    # Insert j in the kth position according to VF(j) = (1/k) sum_(k=1) 1/k (harmonic selection)
    
    sequence = A + B

    candidates = []

    for i in B:
        candidate = vshape_candidates(jobs, A, B, common_due_date, i, len(sequence), True)
        if len(candidate) > 0:
            candidates.append([i, candidate])

    for i in A:
        candidate = vshape_candidates(jobs, A, B, common_due_date, i, len(sequence), True)
        if len(candidate) > 0:
            candidates.append([i, candidate])
    
    moves = []
    if len(candidates) > 0:
        for k in candidates:
            for i in k[1]:
                A_new, B_new = insert_job(A, B, k[0], i)
                f_new = total_cost(jobs, A_new, B_new)
                moves.append([f_new, k[0], i])

    if len(moves) > 0:
        sorted_moves = sorted(moves, key=lambda x: x[0], reverse=True)
        move = harmonic_select(sorted_moves)
        A, B = insert_job(A, B, move[1], move[2])

    return A, B

# ==================================================================
#                       RANDOM PERTURBATION
# ==================================================================

def random_perturbation(jobs, A, B, common_due_date, threshold_inserts, threshold_swaps, insert_probability):
    # Performs randomly selected insert or swap abiding to v-shape
    
    sequence = A + B

    insert = random.random() < insert_probability
    if insert: 
        threshold = threshold_inserts
    else: 
        threshold = threshold_swaps

    choices = sequence[:]
    
    while len(choices) > 0:
        j = random.choice(choices)
        candidates = vshape_candidates(jobs, A, B, common_due_date, j, threshold, insert)
        if len(candidates) > 0:
            i = random.choice(candidates)
            if insert:
                A, B = insert_job(A, B, j, i)
                return A, B
            else:
                A, B = swap_job(A, B, j, i)
            return A, B
        else:
            choices.remove(j)
    return A, B

# ==================================================================
#                     ITERATED LOCAL SEARCH
# ==================================================================

def iterated_local_search(jobs, common_due_date, A, B, threshold_swaps, threshold_inserts, insert_probability, stop_iter, P, Q, tabu_parameters, L_moves):
    start_time = time.time()

    A_best = A[:]
    B_best = B[:]
    best_cost = total_cost(jobs, A_best, B_best)

    iter_no_improv = 0
    iter_count = 0
    while iter_no_improv < stop_iter:
        iter_count += 1

        # Local search phase
        A, B = local_search(jobs, A, B, common_due_date, threshold_swaps, threshold_inserts, insert_probability)
        cost = total_cost(jobs, A, B)

        if cost < best_cost:
            best_cost = cost
            A_best = A[:]
            B_best = B[:]
            iter_no_improv = 0
        else:
            iter_no_improv += 1

        L1, L2, L3 = L_moves

        x = round(random.random(), 1)
        y = round(random.random(), 2)
        
        if x < P:
            A, B, improved = tabu_perturbation(jobs, A, B, common_due_date, tabu_parameters, threshold_inserts, threshold_swaps, insert_probability, best_cost, L1)
            if improved:
                A_best = A
                B_best = B
                best_cost = total_cost(jobs, A, B)
        elif y < (1 - P) * Q:
            for _ in range(L2):
                A, B = construction_perturbation(jobs, A, B, common_due_date)
        else:
            for _ in range(L3):
                A, B = random_perturbation(jobs, A, B, common_due_date, threshold_inserts, threshold_swaps, insert_probability)    
        
    end_time = time.time()
    elapsed_time = end_time - start_time

    return A_best, B_best, round(best_cost, 2), round(elapsed_time, 3), iter_count

# ==================================================================
#                       RUN MAIN FUNCTION
# ==================================================================

def main():
    # TEST CASE
    A = [2, 6, 18, 4]
    #A = [x - 1 for x in A]
    B = [16, 7, 11, 9, 12, 13, 15, 14, 1, 3, 5, 19, 17, 10, 8, 0]
    #B = [x - 1 for x in B]

    maximum_due_date = 0
    for i in A:
        maximum_due_date += jobs[i].processing_time
    for i in B:
        maximum_due_date += jobs[i].processing_time
    common_due_date = maximum_due_date * 0.2
    common_due_date = int(round(common_due_date, 0))    # Common due date for all jobs

    # TEST PARAMETERS
    n = len(A) + len (B)
    threshold_swaps = n//2        # Maximum neighborhood size for swaps
    threshold_inserts = n//3      # Maximum neighborhood size for inserts
    insert_probability = 0.5      # Probability of applying insert move over swap
    stop_iter = 100      # Maximum iterations without improvement to stop
    alpha_1 = 0.5       # Tabu list size deterministic parameter
    alpha_2 = 0.9       # Tabu list size probabilistic parameter
    tabu_parameters = [alpha_1, alpha_2]
    L1 = 10     # Number of moves for tabu perturbation
    L2 = 4      # Number of moves for construction perturbation 
    L3 = n//3   # Number of moves for random perturbation
    L_moves = [L1, L2, L3]
    P = 0.75    # Probability of applying tabu perturbation
    Q = 0.5    # Probability of applying construction over random perturbation
    
    x = total_cost(jobs, A, B)
    ils = iterated_local_search(jobs, common_due_date, A, B, threshold_swaps, threshold_inserts, insert_probability, stop_iter, P, Q, tabu_parameters, L_moves)
    print(f'new_sequence = {ils[0], ils[1]}, best_cost = {ils[2]}, time_count = {ils[3]}, iter_count = {ils[4]}')
    print(f'Melhoria = {100*((x - ils[2])/x)}%')

if __name__ == "__main__":
    main()