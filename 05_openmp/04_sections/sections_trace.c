#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void write_trace_event(FILE *f, int *first, const char *name, double start_s, double end_s, int tid) {
    double ts_us = start_s * 1e6;
    double dur_us = (end_s - start_s) * 1e6;
    if (dur_us < 0.0) dur_us = 0.0;

    if (!*first) fprintf(f, ",\n");
    *first = 0;
    fprintf(f,
            "{\"name\":\"%s\",\"cat\":\"sections\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":1,\"tid\":%d}",
            name, ts_us, dur_us, tid);
}

int main(void) {
    const int N = 1000000;
    int *x = (int *)malloc((size_t)N * sizeof(int));
    long long sum = 0, sum2 = 0;
    long double sum3 = 0.0L;
    int upper = 0, lower = 0;
    const int divide = 20;
    if (!x) {
        fprintf(stderr, "Allocation failed for N=%d\n", N);
        return 1;
    }

    double t_base = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        x[i] = i;
    }

    double t_count_start = 0.0, t_count_end = 0.0;
    double t_sum_start = 0.0, t_sum_end = 0.0;
    double t_sum2_start = 0.0, t_sum2_end = 0.0;
    double t_sum3_start = 0.0, t_sum3_end = 0.0;
    int tid_count = -1, tid_sum = -1, tid_sum2 = -1, tid_sum3 = -1;

    #pragma omp parallel shared(x, sum, sum2, sum3, upper, lower)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                t_count_start = omp_get_wtime();
                tid_count = omp_get_thread_num();
                for (int i = 0; i < N; i++) {
                    if (x[i] > divide) upper++;
                    if (x[i] <= divide) lower++;
                }
                t_count_end = omp_get_wtime();
                printf("The number of points at or below %d in x is %d\n", divide, lower);
                printf("The number of points above %d in x is %d\n", divide, upper);
            }

            #pragma omp section
            {
                t_sum_start = omp_get_wtime();
                tid_sum = omp_get_thread_num();
                for (int i = 0; i < N; i++) {
                    sum = sum + x[i];
                }
                t_sum_end = omp_get_wtime();
                printf("Sum of x = %lld\n", sum);
            }

            #pragma omp section
            {
                t_sum2_start = omp_get_wtime();
                tid_sum2 = omp_get_thread_num();
                for (int i = 0; i < N; i++) {
                    sum2 = sum2 + x[i] * x[i];
                }
                t_sum2_end = omp_get_wtime();
                printf("Sum2 of x = %lld\n", sum2);
            }

            #pragma omp section
            {
                t_sum3_start = omp_get_wtime();
                tid_sum3 = omp_get_thread_num();
                for (int i = 0; i < N; i++) {
                    sum3 = sum3 + (long double)x[i] * x[i] * x[i];
                }
                t_sum3_end = omp_get_wtime();
                printf("Sum3 of x = %.0Lf\n", sum3);
            }
        }
    }

    FILE *trace = fopen("sections_trace.json", "w");
    if (!trace) {
        fprintf(stderr, "Warning: could not open sections_trace.json for writing\n");
        return 0;
    }

    int first = 1;
    fprintf(trace, "{\"traceEvents\":[\n");
    write_trace_event(trace, &first, "count_above_below", t_count_start - t_base, t_count_end - t_base, tid_count);
    write_trace_event(trace, &first, "sum", t_sum_start - t_base, t_sum_end - t_base, tid_sum);
    write_trace_event(trace, &first, "sum2", t_sum2_start - t_base, t_sum2_end - t_base, tid_sum2);
    write_trace_event(trace, &first, "sum3", t_sum3_start - t_base, t_sum3_end - t_base, tid_sum3);
    fprintf(trace, "\n],\"displayTimeUnit\":\"ms\"}\n");
    fclose(trace);

    printf("trace_file=sections_trace.json\n");
    free(x);
    return 0;
}
