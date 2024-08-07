"""
============================================================
Single-Machine Scheduling ILS 
============================================================
Author: Giovanni Cesar Meira Barboza
Date: August 2024
Description:
This script optimizes job scheduling with an Iterated Local Search algorithm. It supports 
evaluating swap and insertion moves, calculating completion times, and modifying job sequences.

============================================================
"""

# ==================================================================
#                     CONFIGURATION
# ==================================================================

import random
import time

# Define Job class

class Job:
    def __init__(self, processing_time, weight_e, weight_t):
        self.processing_time = processing_time
        self.weight_e = weight_e
        self.weight_t = weight_t

# Test values
# Define jobs from a three-column list: [processing_time, weight_e, weight_t]
job_data = [[20, 4, 5], [6, 1, 15], [13, 5, 13], [13, 2, 13], [12, 7, 6], [12, 9, 8], [12, 5, 15], [3, 6, 1], [12, 6, 8], [13, 10, 1]
    # Add more job data as needed
]

# Convert list to Job objects
jobs = [Job(processing_time, weight_e, weight_t) for processing_time, weight_e, weight_t in job_data]


# ==================================================================
#                     AUXILIARY FUNCTIONS
# ==================================================================

# Cummulative time calculation

def cummulative_time(jobs, sequence, sequence_index):
    # Calculates the sum of the processing times from the first job untill sequence_index

    current_time = 0
    for idx in range(sequence_index + 1):
        job = jobs[sequence[idx]]
        current_time += job.processing_time

    return current_time

def total_cost(jobs, sequence, common_due_date):
    A = []  # Early jobs

    current_time = 0
    for x in sequence:
        job = jobs[x]
        current_time += job.processing_time
        if current_time > common_due_date:
            break
        A.append(job)
    
    current_index = sequence.index(x)
    B = [jobs[x] for x in sequence[current_index:]]  # Late jobs

    # Cost B
    cummulative_time_B = 0
    cost_B = 0
    for job in B:
        cummulative_time_B += job.processing_time
        cost_B += cummulative_time_B * job.weight_t
        
    # Cost A
    cummulative_time_A = 0
    cost_A = 0
    A = [*reversed(A)]
    for job in A:
        cost_A += cummulative_time_A * job.weight_e
        cummulative_time_A += job.processing_time

    first_job_time = common_due_date - cummulative_time_A

    return cost_A + cost_B, first_job_time

# Completion time of a job

def completion_time(jobs, sequence, first_job_time, sequence_index):
    current_time = first_job_time
    for idx in range(sequence_index + 1):
        job = jobs[sequence[idx]]
        current_time += job.processing_time

    return current_time

# Weighted penalty calculation

def penalty(job, completion_time, common_due_date):
    if completion_time < common_due_date:
        return job.weight_e * (common_due_date - completion_time)
    else:
        return job.weight_t * (completion_time - common_due_date)

# Insertion and Swap functions

def insert_job(sequence, from_index, to_index):
    job = sequence.pop(from_index)
    if to_index > from_index:
        to_index -= 1
    sequence.insert(to_index, job)
    return sequence

def swap_job(sequence, from_index, to_index):
    # Swap the elements at from_index and to_index
    sequence[from_index], sequence[to_index] = sequence[to_index], sequence[from_index]
    return sequence

# ==================================================================
#                      EVALUATION PROCEDURE
# ==================================================================

def evaluation_procedure(jobs, sequence, common_due_date, j, insert):
    # Require: list of jobs, sequence to be explored, index of the job j to evaluate swaps and insertions possible within the sequence, insert (True -> insert; False -> swap)
    # Ensure: minimum cost of objective function, operation used (swap or insertion), index of the job swapped with or inserted before
    
    #print("evaluation", sequence, j, insert)

    F = []
    cost = total_cost(jobs, sequence, common_due_date)
    f = cost[0]
    first_job_time = cost[1]
    
    Pj = penalty(jobs[sequence[j]], completion_time(jobs, sequence, first_job_time, j), common_due_date)

    for i in range(len(sequence)):
        if i != j: # Avoid auto-insertion/auto-swap
        
            if insert and abs(i - j) != 1:   # INSERTION and avoid swap with the next/previous as an insertion
                Pi = penalty(jobs[sequence[i]], completion_time(jobs, sequence, first_job_time, i), common_due_date)
                new_sequence = sequence[:]
                new_sequence = insert_job(new_sequence, j, i)
                
                k = i           # insert before j
                if j < i: k-=1  # insert after j
            
                new_first_job_time = total_cost(jobs, new_sequence, common_due_date)[1]

                Pi_new = penalty(jobs[sequence[i]], completion_time(jobs, new_sequence, new_first_job_time, k+1), common_due_date) # Penalty value of job i after insertion
                Pj_new = penalty(jobs[sequence[j]], completion_time(jobs, new_sequence, new_first_job_time, k), common_due_date)
                
                # f' = f + (Pi' - Pi) + (Pj' - Pj)
                f_new = f + (Pi_new - Pi) + (Pj_new - Pj)
                F.append([f_new, "i", i])     # (new objective function, i for insertion, index to insert before)
    
            elif not insert:    # SWAP
                new_sequence = sequence[:]
                new_sequence = swap_job(new_sequence, j, i)

                if j > i:   # swap before j
                    l = i
                    k = j
                else:       # swap after j
                    l = j
                    k = i

                new_first_job_time = total_cost(jobs, new_sequence, common_due_date)[1]

            # f' = f + sum^j_{l=i}(Pl'-Pl)
                sum_penalty = 0
                while l <= k:
                    Pi = penalty(jobs[sequence[l]], completion_time(jobs, sequence, first_job_time, l), common_due_date)
                    Pi_new = penalty(jobs[new_sequence[l]], completion_time(jobs, new_sequence, new_first_job_time, l), common_due_date)
                    sum_penalty += (Pi_new - Pi)
                    l += 1
                f_new = f + sum_penalty
                F.append([f_new, "s", i])     # (new objective function, s for swap, index to swap with)

    return min(F, key=lambda x: x[0])

# ==================================================================
#                      LOCAL SEARCH
# ==================================================================

def local_search(jobs, sequence, common_due_date, job_index, threshold_swaps, threshold_inserts, insert_probability):
    # Require: list of jobs, sequence, index of the job in the sequence, number of jobs before and after for swaps and inserts neighborhood, probability of choosing insert 
    # Ensure: sequence with least weighted tardiness/earliness within the neighborhood after all swaps or insertions are made around the chosen job

    original_sequence = sequence[:]

    insert = random.random() >= insert_probability
    if insert:  # Insertion move
        job = sequence[job_index]
        start = max(job_index - threshold_inserts, 0)
        end = min(job_index + threshold_inserts + 1, len(sequence))
        sublist = sequence[start:end]

        sublist_job_index = sublist.index(job)
        move = evaluation_procedure(jobs, sublist, common_due_date, sublist_job_index, True)
        sublist = insert_job(sublist, sublist_job_index, move[2])

        sequence[start:end] = sublist
    
    else:   # Swap move
        job = sequence[job_index]
        start = max(job_index - threshold_swaps, 0)
        end = min(job_index + threshold_swaps + 1, len(sequence))
        sublist = sequence[start:end]

        sublist_job_index = sublist.index(job)
        move = evaluation_procedure(jobs, sublist, common_due_date, sublist_job_index, False)
        sublist = swap_job(sublist, sublist_job_index, move[2])

        sequence[start:end] = sublist
    
    if total_cost(jobs, original_sequence, common_due_date)[0] < total_cost(jobs, sequence, common_due_date)[0]:    # Check whether original solution before move is better
        return(original_sequence)
    else:
        return sequence

# ==================================================================
#                  ITERATED LOCAL SEARCH
# ==================================================================

def perturbation(tabu, n, k):
    # Require: "tabu" list with previous job indexes, lenght of sequence (n), iterations (k) for a job to leave the tabu list
    # Ensure: updated tabu list and job index suggestion for next iteration

    if len(tabu) == k:
        tabu = tabu[1:]
    
    all_indices = list(range(n))
    valid_indices = [i for i in all_indices if i not in tabu]
    if valid_indices:
        job_index = random.choice(valid_indices)
    tabu.append(job_index)

    return tabu, job_index

def iterated_local_search(jobs, common_due_date, initial_sequence, threshold_swaps, threshold_inserts, insert_probability, stop_iter):
    start_time = time.time()

    job_index = random.randint(0, len(initial_sequence) - 1)    # Chooses a random index from the list to start
    iter_no_improv = 0
    best_sequence = initial_sequence[:]
    best_cost = total_cost(jobs, initial_sequence, common_due_date)[0]
    iter_count = 0
    tabu = [job_index]

    sequence = initial_sequence[:]
    while iter_no_improv < stop_iter:
        sequence = local_search(jobs, sequence, common_due_date, job_index, threshold_swaps, threshold_inserts, insert_probability)
        new_total_cost = total_cost(jobs, sequence, common_due_date)
        if new_total_cost[0] < best_cost:
            best_sequence = sequence[:]
            best_cost = new_total_cost[0]
            iter_no_improv = 0
        else:
            iter_no_improv += 1

        iter_count += 1

        # To follow the article, from this point there should be Tabu-based, Construction-based and random perturbations
        # I implemented here my own perturbation, which simply chooses a random job, avoiding to choose the same ones from the previous iterations

        tabu, job_index = perturbation(tabu, len(sequence), 10)     # k = 10 for tabu search is suggested by Qin et al. 2015 
        
        end_time = time.time()
        elapsed_time = end_time - start_time

    return best_sequence, round(best_cost, 2), iter_count, round(elapsed_time, 3)

# ==================================================================
#                       RUN MAIN FUNCTION
# ==================================================================

def main():
    initial_sequence = [4, 2, 1, 3, 7, 9, 6, 5, 8, 10]
    initial_sequence = [x - 1 for x in initial_sequence]
    common_due_date = cummulative_time(jobs, initial_sequence, len(initial_sequence)-1) * 0.8  # Common due date for all jobs
    threshold_swaps = len(initial_sequence)//2
    threshold_inserts = len(initial_sequence)//3
    insert_probability = 0.5
    stop_iter = 50
    ils = iterated_local_search(jobs, common_due_date, initial_sequence, threshold_swaps, threshold_inserts, insert_probability, stop_iter)
    print(f'new_sequence = {ils[0]}, best_cost = {ils[1]}, iter_count = {ils[2]}')
    x = total_cost(jobs,initial_sequence,common_due_date)[0]
    print(f'Melhoria = {x - ils[1]}')

if __name__ == "__main__":
    main()