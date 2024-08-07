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
1. takes in S, job_index and neighborhood (threshold_swap and threshold_insert)
2. randomly decides move (insert or swap)
3. isolate neighborhood around job_index
4. find best move within neighbohood through evaluation_procedure
5. perform swap/insertion on neighborhood
6. reconstruct S with updated neghborhood

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
