#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void write_trace_event(FILE *f, int *first, const char *name,
                              double start_s, double end_s, int tid) {
    double ts_us = start_s * 1e6;
    double dur_us = (end_s - start_s) * 1e6;
    if (dur_us < 0.0) dur_us = 0.0;

    if (!*first) fprintf(f, ",\n");
    *first = 0;
    fprintf(f,
            "{\"name\":\"%s\",\"cat\":\"fib_omp\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":1,\"tid\":%d}",
            name, ts_us, dur_us, tid);
}

static long long fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main(void) {
    const int lo = 20, hi = 40;
    long long *results = (long long *)malloc((hi + 1) * sizeof(long long));
    double *start = (double *)malloc((hi + 1) * sizeof(double));
    double *end = (double *)malloc((hi + 1) * sizeof(double));
    int *tid = (int *)malloc((hi + 1) * sizeof(int));
    if (!results || !start || !end || !tid) {
        free(results);
        free(start);
        free(end);
        free(tid);
        return 1;
    }

    double t_base = omp_get_wtime();
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = lo; i <= hi; i++) {
        start[i] = omp_get_wtime();
        tid[i] = omp_get_thread_num() + 1;
        results[i] = fib(i);
        end[i] = omp_get_wtime();
    }
    double t1 = omp_get_wtime();

    printf("fib_omp: fib(%d)=%lld fib(%d)=%lld\n", lo, results[lo], hi, results[hi]);
    printf("fib_omp_time_s=%.6f\n", t1 - t0);

    FILE *trace = fopen("fib_omp_trace.json", "w");
    if (!trace) {
        fprintf(stderr, "Warning: could not open fib_omp_trace.json for writing\n");
    } else {
        int first = 1;
        fprintf(trace, "{\"traceEvents\":[\n");
        for (int i = lo; i <= hi; i++) {
            char name[32];
            snprintf(name, sizeof(name), "fib(%d)", i);
            write_trace_event(trace, &first, name, start[i] - t_base, end[i] - t_base, tid[i]);
        }
        fprintf(trace, "\n],\"displayTimeUnit\":\"ms\"}\n");
        fclose(trace);
        printf("trace_file=fib_omp_trace.json\n");
    }

    free(results);
    free(start);
    free(end);
    free(tid);
    return 0;
}
