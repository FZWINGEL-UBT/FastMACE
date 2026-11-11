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
 *   2  virtual:  every request gets a reserved virtual address range (cuMemAddressReserve) whose granules all map
 *                one shared physical chunk (cuMemCreate of one granule, cuMemMap at every granule): every pointer
 *                is valid device memory, as Triton's launcher checks at capture, yet the recording costs one
 *                granule. A capture only records pointers; such a graph aliases all its buffers and must never be
 *                replayed. Frees unmap and return the range. Nothing may execute on these ranges: a request from a
 *                stream that is not capturing is refused and counted (arena_uncaptured), and the shared granule is
 *                a canary (arena_alias_fill before the recording, arena_alias_intact after): any kernel that wrote
 *                to the record pool changed it.
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
/* The driver's CUmemAllocationProp / CUmemAccessDesc (cuda.h, CUDA 10.2 on), spelled out: no CUDA headers here. */
typedef struct {
    int type, id;
} mem_location;
typedef struct {
    int type, requested_handle_types;
    mem_location location;
    void *win32_handle_metadata;
    unsigned char compression_type, gpu_direct_rdma_capable;
    unsigned short usage;
    unsigned char reserved[4];
} mem_prop;
typedef struct {
    mem_location location;
    int flags;
} mem_access;
typedef int (*mem_create_t)(unsigned long long *, size_t, const mem_prop *, unsigned long long);
typedef int (*mem_granularity_t)(size_t *, const mem_prop *, int);
typedef int (*mem_map_t)(uint64_t, size_t, size_t, unsigned long long, unsigned long long);
typedef int (*mem_unmap_t)(uint64_t, size_t);
typedef int (*mem_set_access_t)(uint64_t, size_t, const mem_access *, size_t);
typedef int (*mem_release_t)(unsigned long long);
typedef int (*stream_capturing_t)(void *, int *);
typedef int (*memset_d8_t)(uint64_t, unsigned char, size_t);
typedef int (*memcpy_dtoh_t)(void *, uint64_t, size_t);

static mem_alloc_t mem_alloc;
static mem_free_t mem_free;
static mem_info_t mem_info;
static va_reserve_t va_reserve;
static va_free_t va_free;
static mem_create_t mem_create;
static mem_granularity_t mem_granularity;
static mem_map_t mem_map;
static mem_unmap_t mem_unmap;
static mem_set_access_t mem_set_access;
static mem_release_t mem_release;
static stream_capturing_t stream_capturing;
static memset_d8_t memset_d8;
static memcpy_dtoh_t memcpy_dtoh;
static uint64_t alias_probe; /* one granule of address space that maps the shared chunk, for the canary */
static int64_t uncaptured;
static unsigned long long alias_handle; /* the shared physical chunk of mode 2 */
static size_t alias_granule;
static int alias_device = -1;

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
    mem_create = (mem_create_t)dlsym(h, "cuMemCreate");
    mem_granularity = (mem_granularity_t)dlsym(h, "cuMemGetAllocationGranularity");
    mem_map = (mem_map_t)dlsym(h, "cuMemMap");
    mem_unmap = (mem_unmap_t)dlsym(h, "cuMemUnmap");
    mem_set_access = (mem_set_access_t)dlsym(h, "cuMemSetAccess");
    mem_release = (mem_release_t)dlsym(h, "cuMemRelease");
    stream_capturing = (stream_capturing_t)dlsym(h, "cuStreamIsCapturing");
    memset_d8 = (memset_d8_t)dlsym(h, "cuMemsetD8_v2");
    memcpy_dtoh = (memcpy_dtoh_t)dlsym(h, "cuMemcpyDtoH_v2");
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

static int alias_ready(int device) {
    if (alias_handle && alias_device == device) return 0;
    if (alias_handle) return -1; /* one device per process */
    if (!mem_create || !mem_granularity || !mem_map || !mem_unmap || !mem_set_access || !mem_release) return -1;
    mem_prop prop = {0};
    prop.type = 1;          /* CU_MEM_ALLOCATION_TYPE_PINNED */
    prop.location.type = 1; /* CU_MEM_LOCATION_TYPE_DEVICE */
    prop.location.id = device;
    size_t granule = 0;
    if (mem_granularity(&granule, &prop, 0) != 0 || !granule) return -1;
    unsigned long long handle = 0;
    if (mem_create(&handle, granule, &prop, 0) != 0) return -1;
    uint64_t probe = 0;
    mem_access access = {{1, device}, 3};
    if (va_reserve(&probe, granule, granule, 0, 0) != 0) {
        mem_release(handle);
        return -1;
    }
    if (mem_map(probe, granule, 0, handle, 0) != 0 || mem_set_access(probe, granule, &access, 1) != 0) {
        mem_unmap(probe, granule);
        va_free(probe, granule);
        mem_release(handle);
        return -1;
    }
    alias_handle = handle;
    alias_granule = granule;
    alias_device = device;
    alias_probe = probe;
    return 0;
}

static void alias_unmap(uint64_t p, size_t mapped) {
    for (size_t at = 0; at < mapped; at += alias_granule) mem_unmap(p + at, alias_granule);
}

static int capturing(void *stream) {
    int status = 0; /* CU_STREAM_CAPTURE_STATUS_ACTIVE = 1 */
    return stream_capturing && stream_capturing(stream, &status) == 0 && status == 1;
}

static uint64_t virtual_alloc(size_t size, int device, void *stream) {
    uint64_t p = 0;
    if (resolve() != 0 || !va_reserve || virtual_n >= VIRTUAL_CAP || alias_ready(device) != 0) return 0;
    if (!capturing(stream)) { /* memory used outside a capture would run on the aliased granule */
        uncaptured++;
        return 0;
    }
    size_t span = (size + alias_granule - 1) / alias_granule * alias_granule;
    if (va_reserve(&p, span, alias_granule, 0, 0) != 0) return 0;
    size_t at = 0;
    for (; at < span; at += alias_granule)
        if (mem_map(p + at, alias_granule, 0, alias_handle, 0) != 0) break;
    mem_access access = {{1, device}, 3}; /* the device, CU_MEM_ACCESS_FLAGS_PROT_READWRITE */
    if (at < span || mem_set_access(p, span, &access, 1) != 0) {
        alias_unmap(p, at);
        va_free(p, span);
        return 0;
    }
    virtual_ptr[virtual_n] = p;
    virtual_size[virtual_n] = (int64_t)span;
    virtual_n++;
    return p;
}

static int virtual_release(uint64_t p) {
    for (int64_t i = 0; i < virtual_n; i++) {
        if (virtual_ptr[i] == p) {
            alias_unmap(p, (size_t)virtual_size[i]);
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
        p = virtual_alloc(size, device, stream);
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
int64_t arena_uncaptured(void) { return uncaptured; }
/* The canary: fill the shared granule with ``value`` (device must be the pool's), then check it. -1: no granule. */
int arena_alias_fill(int device, int value) {
    if (resolve() != 0 || alias_ready(device) != 0 || !memset_d8) return -1;
    uncaptured = 0;
    return memset_d8(alias_probe, (unsigned char)value, alias_granule) == 0 ? 0 : -1;
}
int arena_alias_intact(int value) {
    if (!alias_probe || !memcpy_dtoh) return -1;
    unsigned char *host = malloc(alias_granule);
    if (!host) return -1;
    int ok = memcpy_dtoh(host, alias_probe, alias_granule) == 0;
    for (size_t i = 0; ok && i < alias_granule; i++) ok = host[i] == (unsigned char)value;
    free(host);
    return ok;
}
uint64_t arena_base(void) { return base; }
int arena_release(void) {
    if (base && resolve() == 0) mem_free(base);
    base = 0;
    if (alias_handle && !virtual_n && mem_release) { /* the shared chunk, once no range maps it */
        mem_unmap(alias_probe, alias_granule);
        va_free(alias_probe, alias_granule);
        alias_probe = 0;
        mem_release(alias_handle);
        alias_handle = 0;
        alias_device = -1;
    }
    return 0;
}
