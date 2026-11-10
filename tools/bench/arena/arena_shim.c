/* Capture-arena allocator shim for torch.cuda.memory.CUDAPluggableAllocator (stage 0 of the capture arena,
 * results/capture-arena-20261110). Plain C, no CUDA headers: device memory comes from the driver API (libcuda,
 * always loaded with a CUDA context), so gcc alone builds it.
 *
 * Modes (arena_mode):
 *   0  record:  every request is served by cuMemAlloc and logged (malloc: size, pointer; free: size, pointer).
 *   1  planned: the k-th malloc is served at base + offset[k] of one arena when its size equals the plan's; any
 *               other request (a mismatch, or past the plan's end) falls back to cuMemAlloc and is counted.
 *               Frees inside the arena are no-ops; the arena is released by arena_release().
 * Every call is logged in both modes, so a planned capture can be checked against its plan.
 */
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef int (*mem_alloc_t)(uint64_t *, size_t);
typedef int (*mem_free_t)(uint64_t);

static mem_alloc_t mem_alloc;
static mem_free_t mem_free;

#define LOG_CAP (1 << 20)
static int64_t log_kind[LOG_CAP], log_size[LOG_CAP];
static uint64_t log_ptr[LOG_CAP];
static int64_t log_n;

static int mode;
static int64_t plan_n, plan_k, plan_total, mismatches, fallback_bytes;
static int64_t *plan_size, *plan_offset;
static uint64_t base;

static int resolve(void) {
    if (mem_alloc && mem_free) return 0;
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) return -1;
    mem_alloc = (mem_alloc_t)dlsym(h, "cuMemAlloc_v2");
    mem_free = (mem_free_t)dlsym(h, "cuMemFree_v2");
    return (mem_alloc && mem_free) ? 0 : -1;
}

static void record(int64_t kind, int64_t size, uint64_t ptr) {
    if (log_n < LOG_CAP) {
        log_kind[log_n] = kind;
        log_size[log_n] = size;
        log_ptr[log_n] = ptr;
    }
    log_n++;
}

static uint64_t raw_alloc(size_t size) {
    uint64_t p = 0;
    if (resolve() != 0 || mem_alloc(&p, size) != 0) return 0;
    return p;
}

void *arena_malloc(size_t size, int device, void *stream) {
    (void)device;
    (void)stream;
    uint64_t p = 0;
    if (mode == 1 && plan_k < plan_n && plan_size[plan_k] == (int64_t)size) {
        if (!base) base = raw_alloc((size_t)plan_total);
        if (base) p = base + (uint64_t)plan_offset[plan_k];
    }
    if (mode == 1) {
        if (!p) {
            mismatches++;
            fallback_bytes += (int64_t)size;
        }
        plan_k++;
    }
    if (!p) p = raw_alloc(size);
    record(1, (int64_t)size, p);
    return (void *)p;
}

void arena_free(void *ptr, size_t size, int device, void *stream) {
    (void)device;
    (void)stream;
    uint64_t p = (uint64_t)ptr;
    record(2, (int64_t)size, p);
    if (base && p >= base && p < base + (uint64_t)plan_total) return;
    if (resolve() == 0) mem_free(p);
}

/* ---- control, through ctypes ---- */
int arena_mode(int m) { mode = m; return mode; }
int64_t arena_log_count(void) { return log_n; }
void arena_log_clear(void) { log_n = 0; }
int64_t arena_log_get(int64_t i, int64_t *kind, int64_t *size, uint64_t *ptr) {
    if (i < 0 || i >= log_n || i >= LOG_CAP) return -1;
    *kind = log_kind[i];
    *size = log_size[i];
    *ptr = log_ptr[i];
    return 0;
}
int arena_set_plan(int64_t n, const int64_t *sizes, const int64_t *offsets, int64_t total) {
    free(plan_size);
    free(plan_offset);
    plan_size = malloc(sizeof(int64_t) * (size_t)(n ? n : 1));
    plan_offset = malloc(sizeof(int64_t) * (size_t)(n ? n : 1));
    if (!plan_size || !plan_offset) return -1;
    for (int64_t i = 0; i < n; i++) {
        plan_size[i] = sizes[i];
        plan_offset[i] = offsets[i];
    }
    plan_n = n;
    plan_k = 0;
    plan_total = total;
    mismatches = 0;
    fallback_bytes = 0;
    return 0;
}
int64_t arena_mismatches(void) { return mismatches; }
int64_t arena_fallback_bytes(void) { return fallback_bytes; }
int64_t arena_planned_served(void) { return plan_k; }
uint64_t arena_base(void) { return base; }
int arena_release(void) {
    if (base && resolve() == 0) mem_free(base);
    base = 0;
    return 0;
}
