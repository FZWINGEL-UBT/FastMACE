// OPT2-T01 gate 1 prototype: the receiver-streaming forward (vendored OEq PR 223 template, see
// fastmace/kernels/oeq_streaming/NOTICE.md) with the folded radial last layer computed inside the kernel.
//
// w[e, slot, c] = sum_k a[e, k] * Wf[slot_start + c, k] is never stored: a block owns R receivers and a slice of
// CS channels, keeps Wf for its slice in shared memory ([K][S][CS], loaded once), stages each receiver's next T
// edges' a rows in shared memory, and every thread (receiver, channel) forms the T x S weights it needs in
// registers before running the template's unrolled path code on them.
{% from "factorized_projected.cuh" import angular_contraction, node_offset with context %}
using int64_t = signed long long;
using int32_t = int;
using scalar_t = {{ scalar }};
using acc_t = {{ scalar }};

constexpr int INPUT_DIM = {{ schedule.input_dim }};
constexpr int EDGE_DIM = {{ schedule.edge_dim }};
constexpr int OUTPUT_DIM = {{ schedule.output_dim }};
constexpr int CHANNEL_DIM = {{ schedule.channels }};
constexpr int K = {{ k }};
constexpr int CS = {{ cs }};
constexpr int R = {{ r }};
constexpr int T = {{ t }};
{%- set ns = namespace(slots=[]) %}
{%- for path in schedule.paths %}{% for edge_channel in range(path.edge_mul) %}
{%- set ns.slots = ns.slots + [path.weight_start + edge_channel * schedule.channels] %}
{%- endfor %}{% endfor %}
constexpr int S = {{ ns.slots | length }};
__device__ __constant__ int SLOT_START[S] = { {{ ns.slots | join(", ") }} };

extern "C" __global__ void __launch_bounds__(R * CS) fused_projected_forward(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ sh,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ wf,
    const int32_t* __restrict__ senders,
    const int32_t* __restrict__ row_ptr,
    scalar_t* __restrict__ out) {

    extern __shared__ scalar_t smem[];
    scalar_t* wf_s = smem;                  // [K][S][CS]
    scalar_t* a_s = smem + K * S * CS;      // [R][T][K]
    __shared__ int max_chunks;

    const int tid = threadIdx.x;
    const int rl = tid / CS;
    const int cl = tid - rl * CS;
    const int channel = int(blockIdx.y) * CS + cl;
    const int64_t node = int64_t(blockIdx.x) * R + rl;
    const bool active = node < node_count;

    for (int i = tid; i < K * S * CS; i += R * CS) {
        const int k = i / (S * CS);
        const int rest = i - k * (S * CS);
        const int s = rest / CS;
        const int c = rest - s * CS;
        wf_s[i] = wf[int64_t(SLOT_START[s] + int(blockIdx.y) * CS + c) * K + k];
    }
    if (tid == 0) max_chunks = 0;
    __syncthreads();

    int64_t begin = 0, end = 0;
    if (active) {
        begin = row_ptr[node] < 0 ? 0 : row_ptr[node];
        end = row_ptr[node + 1] < begin ? begin : row_ptr[node + 1];
        end = end > edge_count ? edge_count : end;
        if (cl == 0) atomicMax(&max_chunks, int((end - begin + T - 1) / T));
    }
    const int64_t output_base = node * OUTPUT_DIM;

    {%- for slot in schedule.output_slots %}
        scalar_t {{ slot.name }} = scalar_t(0);
    {%- endfor %}
    __syncthreads();
    const int chunks = max_chunks;

    for (int chunk = 0; chunk < chunks; ++chunk) {
        if (chunk) __syncthreads();
        for (int i = tid; i < R * T * K; i += R * CS) {
            const int r = i / (T * K);
            const int rest = i - r * (T * K);
            const int t = rest / K;
            const int k = rest - t * K;
            const int64_t n = int64_t(blockIdx.x) * R + r;
            scalar_t v = scalar_t(0);
            if (n < node_count) {
                int64_t b = row_ptr[n] < 0 ? 0 : row_ptr[n];
                int64_t f = row_ptr[n + 1] < b ? b : row_ptr[n + 1];
                f = f > edge_count ? edge_count : f;
                const int64_t e = b + int64_t(chunk) * T + t;
                if (e < f) v = a[e * K + k];
            }
            a_s[i] = v;
        }
        __syncthreads();
        if (!active) continue;

        acc_t w[T][S];
        #pragma unroll
        for (int t = 0; t < T; ++t)
            #pragma unroll
            for (int s = 0; s < S; ++s) w[t][s] = acc_t(0);
        #pragma unroll 5
        for (int k = 0; k < K; ++k) {
            scalar_t wk[S];
            #pragma unroll
            for (int s = 0; s < S; ++s) wk[s] = wf_s[(k * S + s) * CS + cl];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                const scalar_t av = a_s[(rl * T + t) * K + k];
                #pragma unroll
                for (int s = 0; s < S; ++s) w[t][s] += av * wk[s];
            }
        }

        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const int64_t e = begin + int64_t(chunk) * T + t;
            if (e >= end) break;
            const int32_t sender = senders[e];
            if (sender < 0 || sender >= node_count) continue;
            const int64_t input_base = int64_t(sender) * INPUT_DIM;
            const int64_t edge_base = e * EDGE_DIM;
            {%- set sn = namespace(slot=0) %}
            {% for path in schedule.paths %}
                {% for edge_channel in range(path.edge_mul) %} {
                    const scalar_t radial_weight = w[t][{{ sn.slot }}];
                    {% for output in path.outputs %} {
                        {{ angular_contraction("angular_value", path, output, edge_channel) }}
                        {{ output.accumulator.name }} += radial_weight * angular_value;
                    } {%- endfor %}
                } {%- set sn.slot = sn.slot + 1 %}{%- endfor %}
            {%- endfor %}
        }
    }
    if (!active) return;
    {% for slot in schedule.output_slots %} {
        const int64_t output_index =
            output_base + {{ node_offset(slot.start, slot.component, slot.irrep_dim) }};
        out[output_index] = {{ slot.name }};
    } {%- endfor %}
}
