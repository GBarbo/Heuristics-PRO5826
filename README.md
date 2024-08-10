# Heuristics-PRO5826

Heuristic developed to tackle the Single Machine Penalized Scheduling Problem, according to the Biskup's benchmark (see the report)
Local Search

Main idea:

Iterated Local Search
1.  takes in initial solution S'
2.  randomnly chooses a job to initiate search
3.  iter_no_improv -> 0
4.  while iter_no_improv < stop_iter do:
5.     S = local_search(S', job_index)
6.     if S is better than Sbest:
7.        Sbest = S
8.        iter_no_improv = 0
9.     else:
10.       iter_no_improv += 1
11.    endif
12.    job_index = perturbation(S)
13. endwhile
14. return Sbest 

Local Search
1.  takes in S, job_index and threshold (for swap and insert)
2.  randomly decides move (insert or swap)
3.  defines neighborhood throug vshape_neighborhood(S, threshold, j_index)
4.  find best move within neighbohood through evaluation_procedure
5.  perform swap/insertion on neighborhood

V-Shape Neighborhood
1.  takes in S, threshold and job_index
2.  divides S in subsets of early jobs (A) and late jobs (B)
3.  determines whether job_index belongs to A or B
4.  select neighborhood in the opposite subset of job_index
5.  for each element i of the opposite subset do:
6.      calculate pi/ai if opp. subset = A or pi/bi if B
7.      add to the neighborhood jobs if pi/ai or pi/bi <= pj/aj or pj/bj, j = job_index
8.      if swap is selected:
9           remove from neighborhood if (pi/bi > pj-1/bj-1 or pi/bi < pj+1/bj+1 (if opp. subset = A)) or (pi/ai > pj-1/aj-1 or pi/ai < pj+1/aj+1 (if B))
10. return neighborhood whose swaps/inserts wont disobey v-shape

Perturbation
1. takes in S and number of iterations to leave tabu list (k)
2. if size(S) > k:
3.    remove first job from tabu list
4. endif
5. add last job to tabu list
6. Scopy = hardcopy(S)
7. Scopy = Scopy - tabu
8. randomly chooses job_index in Scopy
9. return new job_index
