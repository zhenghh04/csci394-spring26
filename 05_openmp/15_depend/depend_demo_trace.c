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
            "{\"name\":\"%s\",\"cat\":\"depend\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":1,\"tid\":%d}",
            name, ts_us, dur_us, tid);
}

static void write_flow_event(FILE *f, int *first, const char *name, char ph,
                             char bp, double ts_s, int tid, int flow_id) {
    double ts_us = ts_s * 1e6;
    if (!*first) fprintf(f, ",\n");
    *first = 0;
    fprintf(f,
            "{\"name\":\"%s\",\"cat\":\"depend_flow\",\"ph\":\"%c\",\"bp\":\"%c\",\"ts\":%.3f,\"pid\":1,\"tid\":%d,\"id\":%d}",
            name, ph, bp, ts_us, tid, flow_id);
}

/*
 * Same dependency structure as depend_demo.c, plus trace capture.
 * stage 1: produce A[i]
 * stage 2: consume A[i], produce B[i]
 * stage 3: consume B[i], produce C[i]
 */
int main(void) {
    const int n = 12;
    int *a = (int *)malloc((size_t)n * sizeof(int));
    int *b = (int *)malloc((size_t)n * sizeof(int));
    int *c = (int *)malloc((size_t)n * sizeof(int));
    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }

    double *s1_start = (double *)calloc((size_t)n, sizeof(double));
    double *s1_end = (double *)calloc((size_t)n, sizeof(double));
    int *s1_tid = (int *)calloc((size_t)n, sizeof(int));

    double *s2_start = (double *)calloc((size_t)n, sizeof(double));
    double *s2_end = (double *)calloc((size_t)n, sizeof(double));
    int *s2_tid = (int *)calloc((size_t)n, sizeof(int));

    double *s3_start = (double *)calloc((size_t)n, sizeof(double));
    double *s3_end = (double *)calloc((size_t)n, sizeof(double));
    int *s3_tid = (int *)calloc((size_t)n, sizeof(int));

    if (!s1_start || !s1_end || !s1_tid ||
        !s2_start || !s2_end || !s2_tid ||
        !s3_start || !s3_end || !s3_tid) {
        fprintf(stderr, "trace array allocation failed\n");
        free(a); free(b); free(c);
        free(s1_start); free(s1_end); free(s1_tid);
        free(s2_start); free(s2_end); free(s2_tid);
        free(s3_start); free(s3_end); free(s3_tid);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        a[i] = 0;
        b[i] = 0;
        c[i] = 0;
    }

    double t_base = omp_get_wtime();
    double t0 = omp_get_wtime();

#pragma omp parallel
    {
#pragma omp single
        {
            for (int i = 0; i < n; i++) {
#pragma omp task depend(out : a[i]) firstprivate(i)
                {
                    s1_start[i] = omp_get_wtime();
                    s1_tid[i] = omp_get_thread_num() + 1;
                    a[i] = i + 1;
                    printf("T%d produce a[%d]=%d\n", omp_get_thread_num(), i, a[i]);
                    s1_end[i] = omp_get_wtime();
                }

#pragma omp task depend(in : a[i]) depend(out : b[i]) firstprivate(i)
                {
                    s2_start[i] = omp_get_wtime();
                    s2_tid[i] = omp_get_thread_num() + 1;
                    b[i] = a[i] * 10;
                    printf("T%d transform b[%d]=%d\n", omp_get_thread_num(), i, b[i]);
                    s2_end[i] = omp_get_wtime();
                }

#pragma omp task depend(in : b[i]) depend(out : c[i]) firstprivate(i)
                {
                    s3_start[i] = omp_get_wtime();
                    s3_tid[i] = omp_get_thread_num() + 1;
                    c[i] = b[i] + 7;
                    printf("T%d finalize c[%d]=%d\n", omp_get_thread_num(), i, c[i]);
                    s3_end[i] = omp_get_wtime();
                }
            }
        }
    }

    double t1 = omp_get_wtime();

    long long sum = 0;
    int ok = 1;
    for (int i = 0; i < n; i++) {
        const int expected = (i + 1) * 10 + 7;
        sum += c[i];
        if (c[i] != expected) {
            ok = 0;
            fprintf(stderr, "mismatch at i=%d: got %d expected %d\n", i, c[i], expected);
        }
    }

    printf("check: %s, sum(c)=%lld\n", ok ? "PASS" : "FAIL", sum);
    printf("elapsed_s=%.6f\n", t1 - t0);

    FILE *trace = fopen("depend_demo_trace.json", "w");
    if (!trace) {
        fprintf(stderr, "Warning: could not open depend_demo_trace.json for writing\n");
    } else {
        int first = 1;
        fprintf(trace, "{\"traceEvents\":[\n");
        for (int i = 0; i < n; i++) {
            char name1[32], name2[32], name3[32];
            snprintf(name1, sizeof(name1), "produce(%d)", i);
            snprintf(name2, sizeof(name2), "transform(%d)", i);
            snprintf(name3, sizeof(name3), "finalize(%d)", i);
            write_trace_event(trace, &first, name1, s1_start[i] - t_base, s1_end[i] - t_base, s1_tid[i]);
            write_trace_event(trace, &first, name2, s2_start[i] - t_base, s2_end[i] - t_base, s2_tid[i]);
            write_trace_event(trace, &first, name3, s3_start[i] - t_base, s3_end[i] - t_base, s3_tid[i]);

            /* One flow chain per i: produce(i) -> transform(i) -> finalize(i). */
            write_flow_event(trace, &first, "dep chain", 's', 'e',
                             s1_end[i] - t_base, s1_tid[i], 2*i);
            write_flow_event(trace, &first, "dep chain", 'f', 'e',
                             s2_start[i] - t_base, s2_tid[i], 2*i);
            write_flow_event(trace, &first, "dep chain", 's', 'e',
                             s2_end[i] - t_base, s2_tid[i], 2*i+1);                             
            write_flow_event(trace, &first, "dep chain", 'f', 'e',
                             s3_start[i] - t_base, s3_tid[i], 2*i+1);
        }
        fprintf(trace, "\n],\"displayTimeUnit\":\"ms\"}\n");
        fclose(trace);
        printf("trace_file=depend_demo_trace.json\n");
    }

    free(a); free(b); free(c);
    free(s1_start); free(s1_end); free(s1_tid);
    free(s2_start); free(s2_end); free(s2_tid);
    free(s3_start); free(s3_end); free(s3_tid);
    return ok ? 0 : 2;
}
