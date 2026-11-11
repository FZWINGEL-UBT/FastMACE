/* Capture-arena allocator for torch.cuda.memory.CUDAPluggableAllocator (EngineConfig.capture_arena; stage 0 in
 * results/capture-arena-20261110, the lab copy in tools/bench/arena/arena_shim.c). Plain C, no CUDA headers:
 * device memory and virtual address ranges come from the driver API (libcuda, loaded with any CUDA context), so gcc
 * alone builds it.
 *
 * Modes (arena_mode):
 *   0  record:   every request is served by cuMemAlloc and logged.
 *   1  planned:  the k-th malloc is served at base + offset[k] of one arena when its size equals the plan's; any
 *                other request (a mismatch, or past the plan's end) falls back to cuMemAlloc and is counted.
 *                Frees inside the arena are no-ops; arena_release() returns the arena.
 *   2  virtual:  every request gets a reserved virtual address range with no memory behind it (cuMemAddressReserve).
 *                A capture only records pointers, so a recording capture needs no device memory; such a graph
 *                must never be replayed. Frees return the range.
 * Every call is logged, so a planned capture can be checked against its plan.
 *
 * Guard (owner, 10 November 2026; the HYPERVISOR_ERROR crashes of the local 5060 Ti under WSL2): a request for
 * device memory beyond the device's free memory (cuMemGetInfo) minus the headroom (default 1 GiB) is refused with
 * a null pointer, which PyTorch raises as an out-of-memory error, before the driver is asked. Refusals are counted.
 */
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef int (*mem_alloc_t)(uint64_t *, size_t);
typedef int (*mem_free_t)(uint64_t);
typedef int (*mem_info_t)(size_t *, size_t *);
typedef int (*va_reserve_t)(uint64_t *, size_t, size_t, uint64_t, unsigned long long);
typedef int (*va_free_t)(uint64_t, size_t);

static mem_alloc_t mem_alloc;
static mem_free_t mem_free;
static mem_info_t mem_info;
static va_reserve_t va_reserve;
static va_free_t va_free;

#define LOG_CAP (1 << 20)
static int64_t log_kind[LOG_CAP], log_size[LOG_CAP];
static uint64_t log_ptr[LOG_CAP];
static int64_t log_n;

static int mode;
static int64_t plan_n, plan_k, plan_total, mismatches, fallback_bytes, refusals;
static int64_t *plan_size, *plan_offset;
static uint64_t base;
static int64_t headroom = (int64_t)1 << 30;

#define VIRTUAL_CAP 65536
static uint64_t virtual_ptr[VIRTUAL_CAP];
static int64_t virtual_size[VIRTUAL_CAP];
static int64_t virtual_n;

static int resolve(void) {
    if (mem_alloc && mem_free && mem_info) return 0;
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) return -1;
    mem_alloc = (mem_alloc_t)dlsym(h, "cuMemAlloc_v2");
    mem_free = (mem_free_t)dlsym(h, "cuMemFree_v2");
    mem_info = (mem_info_t)dlsym(h, "cuMemGetInfo_v2");
    va_reserve = (va_reserve_t)dlsym(h, "cuMemAddressReserve");
    va_free = (va_free_t)dlsym(h, "cuMemAddressFree");
    return (mem_alloc && mem_free && mem_info) ? 0 : -1;
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
    size_t free_bytes = 0, total_bytes = 0;
    if (resolve() != 0 || mem_info(&free_bytes, &total_bytes) != 0) return 0;
    if ((int64_t)size > (int64_t)free_bytes - headroom) {
        refusals++;
        return 0;
    }
    if (mem_alloc(&p, size) != 0) return 0;
    return p;
}

static uint64_t virtual_alloc(size_t size) {
    uint64_t p = 0;
    if (resolve() != 0 || !va_reserve || virtual_n >= VIRTUAL_CAP) return 0;
    if (va_reserve(&p, size, (size_t)2 << 20, 0, 0) != 0) return 0;
    virtual_ptr[virtual_n] = p;
    virtual_size[virtual_n] = (int64_t)size;
    virtual_n++;
    return p;
}

static int virtual_release(uint64_t p) {
    for (int64_t i = 0; i < virtual_n; i++) {
        if (virtual_ptr[i] == p) {
            if (va_free) va_free(p, (size_t)virtual_size[i]);
            virtual_n--;
            virtual_ptr[i] = virtual_ptr[virtual_n];
            virtual_size[i] = virtual_size[virtual_n];
            return 1;
        }
    }
    return 0;
}

void *arena_malloc(size_t size, int device, void *stream) {
    (void)device;
    (void)stream;
    uint64_t p = 0;
    if (mode == 2) {
        p = virtual_alloc(size);
        record(1, (int64_t)size, p);
        return (void *)p;
    }
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
    if (virtual_release(p)) return;
    if (base && p >= base && p < base + (uint64_t)plan_total) return;
    if (resolve() == 0) mem_free(p);
}

/* ---- control, through ctypes ---- */
int arena_mode(int m) { mode = m; return mode; }
int64_t arena_set_headroom(int64_t bytes) { headroom = bytes; return headroom; }
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
    if (base) return -2; /* the previous arena is still held */
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
int64_t arena_refusals(void) { return refusals; }
int64_t arena_virtual_live(void) { return virtual_n; }
uint64_t arena_base(void) { return base; }
int arena_release(void) {
    if (base && resolve() == 0) mem_free(base);
    base = 0;
    return 0;
}
