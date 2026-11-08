"""CPU-only arithmetic model. These are explicit-formulation rooflines, NOT proven
algebraic complexity bounds or measured cache-bandwidth values. No GPU APIs.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PROFILE = ROOT/'results/h200-fastpath-20261108/round4/breakdown'
model = json.loads((OUT/'model.json').read_text())
profile = json.loads((PROFILE/'summary.json').read_text())
ops = list(csv.DictReader((PROFILE/'ops.csv').open()))
N,E,C,K,D = 5000,373760,128,64,16
p = model['interactions'][1]
nz = p['CG_nonzero_total']
out = p['output_components_per_channel']
unique = p['unique_input_SH_component_products']
adds = p['within_output_sum_adds']

# Exact semantic counts of the three dominant current dense node chains.
gemm = {'compose_both': 2*(2*N*16*(3*K)*C),
        'interaction2_linear_both': 2*(2*N*C*(256+3*384+5*384+7*256)),
        'interaction2_up_both': 2*(2*N*4*C*C)}
# CG fixed coefficient multiplies, cross-path compiler factoring and special
# unit/zeroSH terms are not fully modeled. This counts one transparent schedule.
forward_ops_channel = unique + adds + 2*out + 6*len(p['paths'])
vec = [q for q in p['paths'] if q['l1'] == 1]
scalar = [q for q in p['paths'] if q['l1'] == 0]
input_components = sum(2*q['l1']+1 for q in p['paths'])
vec_edge_components = sum(2*q['l2']+1 for q in vec)
scalar_edge_components = sum(2*q['l2']+1 for q in scalar)
u = 2*nz-input_components
v = 2*sum(q['CG_nonzero'] for q in vec)-vec_edge_components
wg = 2*input_components-len(p['paths'])
dx = 2*input_components
dsh = 2*vec_edge_components + 2*scalar_edge_components + len(scalar)
dr = 2*len(p['paths'])
radial = 13*len(p['paths'])
work = {'i2_forward': E*C*forward_ops_channel,
        'i2_backward': E*C*(u+v+wg+dx+dsh+dr+radial),
        'i1_forward': 2*E*(D*K+6*K),
        'i1_backward': 2*E*(10*K+2*D*K+K)}
gpu = {'H200': {'fp32_TF': 67, 'tf32_dense_TF': 494.5, 'hbm_TB':4.8},
       'A100_SXM80': {'fp32_TF':19.5,'tf32_dense_TF':156,'hbm_TB':2.039}}
for name, hw in gpu.items():
    hw['tf32x3_semantic_TF'] = hw['tf32_dense_TF']/3
    hw['explicit_work_ideal_us'] = {k:v/(hw['fp32_TF']*1e6) for k,v in work.items()}
    if name == 'A100_SXM80':
        # A100 keeps POWER_BACKWARD off:6-term value+5-term derivative
        # Clenshaw (~33ops/channel vs20for dualHorner), excluding sharedscale.
        hw['explicit_work_ideal_us']['i1_backward'] += E*K*13/(hw['fp32_TF']*1e6)
    hw['dominant_gemm_ideal_us'] = sum(gemm.values())/(hw['tf32x3_semantic_TF']*1e6)
    hw['explicit_work_sum_ideal_us_excluding_SC_geometry'] = sum(hw['explicit_work_ideal_us'].values())+hw['dominant_gemm_ideal_us']
bytes_ = {'x_node': N*512*4, 'raw_message':N*5120*4, 'projected_message':N*2048*4,
          'first_outer_S':N*3*16*64*4, 'edge_SH':E*16*4,
          'positions_and_forces':N*3*4*2,
          'one_requested_spline_pass':E*1280*4*4,
          'spline_allocation':9*731*1280*4*4,
          'weight_and_slope_tape':E*1280*2*4,
          'raw_message_fwd_write_bwd_read':N*5120*4*2,
          'dx_atomic_payload_no_RMW_assumption':E*512*4,
          'SH_dr_backward_output':E*(16+1)*4,
          'dominant_node_GEMM_min_A_C_accesses':2*N*16*(192+128)*4 + 2*N*(5120+2048)*4 + 4*N*512*4,
          'SC_separate_kernel_input_output_accesses':6*N*2048*4+2*N*(512+128)*4}
for name, hw in gpu.items():
    hw['bytes_over_rated_HBM_us_NOT_physical_floor'] = {k:v/(hw['hbm_TB']*1e6) for k,v in bytes_.items()}
    hw['two_pass_logical_spline_HBM_us_NOT_floor'] = 2*bytes_['one_requested_spline_pass']/(hw['hbm_TB']*1e6)

tf = [r for r in ops if r['kernel'].startswith('fm_tf32x3_')]
counts = {'i2_forward_ops_channel': forward_ops_channel,
          'i2_backward_ops_channel_parts':dict(u=u,v=v,wg=wg,dx=dx,dsh=dsh,dr=dr,radial=radial)}
# Spatial-screen examples: illustrative FFT grids, not claimed accuracy settings.
L=39.05
fft = []
for h in (.5,.25):
    m=math.ceil(L/h)
    cells=m**3
    fft.append(dict(spacing_A=h,m=m,grid_cells=cells,
                    input_1536_real_field_bytes=cells*1536*4,
                    output_15360_real_field_bytes=cells*15360*4,
                    scalar_real_FFT_work_proxy_1536=5*cells*math.log2(cells)*1536))
rho=N/L**3
edge_pair_time_us=1905.4
step_us=profile['wall_ms_per_step']*1000
result=dict(source_commit='0d0eefaebf7d1e0152a8cdf80f2cb89f6239b284',profile_commit=profile['fastmace_commit'],
            N=N,E_padded=E,channels=C,counts=counts,direct_work_estimates_flop=work,
            dominant_semantic_node_gemm_flop=gemm,gpu=gpu,array_or_logical_bytes=bytes_,
            measured_H200=dict(step_us=step_us,kernel_us=profile['eager_force_kernel_ms_per_call']*1000,
                               i2_forward_us=806.4,i2_backward_us=1099.0,
                               all_tf32_us=sum(float(r['us_per_call']) for r in tf),
                               all_tf32_launches=sum(float(r['launches_per_call']) for r in tf)),
            measured_A100=dict(step_us=7360,current_kernel_profile_available=False),
            profile_scope=dict(eager_kernels=profile['eager_kernels_per_call'],
                               replay_kernels=profile['force_graph_replay_profile']['kernels_per_replay'],
                               replay_gaps_us=profile['force_graph_replay_profile']['gaps_inside_replays']['total_gap_us_per_replay'],
                               rebuild_calls=profile['stage_calls']['Neighbour-list rebuild (GPU)'],
                               instrumented_steps=profile['args']['steps'],
                               rebuild_us_per_step_already_amortized=1000*profile['stage_ms_per_step']['Neighbour-list rebuild (GPU)'],
                               rebuild_us_per_call=1000*profile['stage_ms_per_step']['Neighbour-list rebuild (GPU)']*profile['args']['steps']/profile['stage_calls']['Neighbour-list rebuild (GPU)'],
                               stage_durations_can_overlap=True),
            thresholds=dict(H200_twenty_percent_us=step_us*.2,H200_twofold_target_us=step_us/2,
                            i2_reduction_for_twenty_percent=step_us*.2/edge_pair_time_us,
                            i2_time_allowed_for_twofold_if_rest_fixed_us=step_us/2-(step_us-edge_pair_time_us)),
            spatial_examples=dict(box_A=L,density_atoms_A3=rho,FFT_grids=fft),
            cache_floor_formula='max(F_SIMT/P_SIMT,3*F_MMA/P_TF32,B_HBM/BW_HBM,B_L2/BW_L2); physical cache bytes/BW_L2 unknown',
            limitations=['Arithmetic estimates omit fixed CG scale cost and some factoring; not irreducible FLOPs.',
                         'Symmetric-contraction operation count, physical DRAM/L2 sectors and calibratedL2BW unavailable.',
                         'Kernel-boundary arrays can be avoided by a different execution plan; not unavoidable model traffic.',
                         'Actualactiveedges may be below paddedE; counts based onpaddedE are upper work budgets.',
                         'A100I1backwardreferenceadjustedfornonPOWERClenshaw; stillapproximateoperationcounts.',
                         'A100has no currentpost-ten kernelprofile; no scaledH200times are measurements.'],
            source_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest()
                           for x in (PROFILE/'ops.csv',PROFILE/'summary.json',ROOT/'docs/optimisation-ledger.md')})
(OUT/'floor-model.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('counts','direct_work_estimates_flop','dominant_semantic_node_gemm_flop','thresholds')},indent=2))
print(json.dumps({k:{kk:vv for kk,vv in v.items() if not kk.startswith('bytes_')} for k,v in gpu.items()},indent=2))
