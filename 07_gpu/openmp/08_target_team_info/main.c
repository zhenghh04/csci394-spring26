#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static int parse_int_arg(char **argv, int index, int default_value) {
    if (argv[index] == NULL) {
        return default_value;
    }
    return atoi(argv[index]);
}

int main(int argc, char **argv) {
    int n = 32;
    int requested_num_teams = 0;
    int requested_thread_limit = 0;

    if (argc >= 2) {
        n = parse_int_arg(argv, 1, n);
    }
    if (argc >= 3) {
        requested_num_teams = parse_int_arg(argv, 2, requested_num_teams);
    }
    if (argc >= 4) {
        requested_thread_limit = parse_int_arg(argv, 3, requested_thread_limit);
    }

    if (n <= 0) {
        fprintf(stderr, "n must be positive\n");
        return 1;
    }

    int *owner_team = malloc((size_t)n * sizeof(int));
    int *owner_thread = malloc((size_t)n * sizeof(int));
    int *owner_num_threads = malloc((size_t)n * sizeof(int));
    if (owner_team == NULL || owner_thread == NULL || owner_num_threads == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(owner_team);
        free(owner_thread);
        free(owner_num_threads);
        return 1;
    }

    int actual_num_teams = -1;
    int threads_in_team0 = -1;
    int ran_on_host = 1;

    if (requested_num_teams > 0 && requested_thread_limit > 0) {
#pragma omp target teams distribute parallel for num_teams(requested_num_teams) thread_limit(requested_thread_limit) map(from : owner_team[0:n], owner_thread[0:n], owner_num_threads[0:n], actual_num_teams, threads_in_team0, ran_on_host)
        for (int i = 0; i < n; i++) {
            owner_team[i] = omp_get_team_num();
            owner_thread[i] = omp_get_thread_num();
            owner_num_threads[i] = omp_get_num_threads();

            if (i == 0) {
                actual_num_teams = omp_get_num_teams();
                threads_in_team0 = omp_get_num_threads();
                ran_on_host = omp_is_initial_device();
            }
        }
    } else if (requested_num_teams > 0) {
#pragma omp target teams distribute parallel for num_teams(requested_num_teams) map(from : owner_team[0:n], owner_thread[0:n], owner_num_threads[0:n], actual_num_teams, threads_in_team0, ran_on_host)
        for (int i = 0; i < n; i++) {
            owner_team[i] = omp_get_team_num();
            owner_thread[i] = omp_get_thread_num();
            owner_num_threads[i] = omp_get_num_threads();

            if (i == 0) {
                actual_num_teams = omp_get_num_teams();
                threads_in_team0 = omp_get_num_threads();
                ran_on_host = omp_is_initial_device();
            }
        }
    } else if (requested_thread_limit > 0) {
#pragma omp target teams distribute parallel for thread_limit(requested_thread_limit) map(from : owner_team[0:n], owner_thread[0:n], owner_num_threads[0:n], actual_num_teams, threads_in_team0, ran_on_host)
        for (int i = 0; i < n; i++) {
            owner_team[i] = omp_get_team_num();
            owner_thread[i] = omp_get_thread_num();
            owner_num_threads[i] = omp_get_num_threads();

            if (i == 0) {
                actual_num_teams = omp_get_num_teams();
                threads_in_team0 = omp_get_num_threads();
                ran_on_host = omp_is_initial_device();
            }
        }
    } else {
#pragma omp target teams distribute parallel for map(from : owner_team[0:n], owner_thread[0:n], owner_num_threads[0:n], actual_num_teams, threads_in_team0, ran_on_host)
        for (int i = 0; i < n; i++) {
            owner_team[i] = omp_get_team_num();
            owner_thread[i] = omp_get_thread_num();
            owner_num_threads[i] = omp_get_num_threads();

            if (i == 0) {
                actual_num_teams = omp_get_num_teams();
                threads_in_team0 = omp_get_num_threads();
                ran_on_host = omp_is_initial_device();
            }
        }
    }

    printf("OpenMP target teams/distribute/parallel-for inspection\n");
    printf("n=%d requested_num_teams=%d requested_thread_limit=%d\n",
           n, requested_num_teams, requested_thread_limit);
    printf("omp_is_initial_device=%d\n", ran_on_host);
    printf("actual_num_teams=%d threads_in_team0=%d\n",
           actual_num_teams, threads_in_team0);
    printf("\n");
    printf("i  team  thread  threads_in_team\n");
    for (int i = 0; i < n; i++) {
        printf("%2d %5d %7d %16d\n",
               i, owner_team[i], owner_thread[i], owner_num_threads[i]);
    }

    free(owner_team);
    free(owner_thread);
    free(owner_num_threads);
    return 0;
}
