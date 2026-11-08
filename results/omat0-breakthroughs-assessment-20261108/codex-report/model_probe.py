"""Small CPU-only metadata/rank probe of the user-specified local checkpoint.

No downloads, suites, full model execution, GPU calls, or source changes.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
import fastmace
import torch
with torch.serialization.safe_globals([slice]):
    from e3nn import o3
from fastmace.rewrites import radial
from fastmace.rewrites.generic import modules as gm

torch.set_num_threads(1)
CHECKPOINT = Path('/home/fzwin/models/mace-omat-0-medium.model')
SHA256 = 'd4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a'
digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
assert digest == SHA256, digest
model = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)


def rank_info(matrix):
    s = torch.linalg.svdvals(matrix.double())
    return {'shape': list(matrix.shape), 'rank_relative_1e12': int((s > s[0]*1e-12).sum()),
            'smax': s[0].item(), 'smin': s[-1].item(),
            'relative_smin': (s[-1]/s[0]).item()}


result = {'checkpoint_sha256': digest, 'checkpoint': str(CHECKPOINT),
          'torch': torch.__version__, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
          'model_class': type(model).__name__, 'r_max_A': float(model.r_max),
          'num_interactions': len(model.interactions), 'atomic_numbers': model.atomic_numbers.tolist(),
          'parameters': sum(p.numel() for p in model.parameters()),
          'radial_embedding_class': type(model.radial_embedding).__name__,
          'radial_embedding_repr': str(model.radial_embedding),
          'node_embedding_repr': str(model.node_embedding),
          'interactions': [], 'products': [], 'readouts': [], 'node_linear_matrices': {}}
result['radial_hyperparameters'] = {'num_bessel': int(model.radial_embedding.bessel_fn.bessel_weights.numel()),
    'cutoff_p': float(model.radial_embedding.cutoff_fn.p),
    'Agnesi_a': float(model.radial_embedding.distance_transform.a),
    'Agnesi_q': float(model.radial_embedding.distance_transform.q),
    'Agnesi_p': float(model.radial_embedding.distance_transform.p)}
for index, interaction in enumerate(model.interactions):
    hidden, last = gm.fc_split(radial._radial_net(interaction))
    w = gm.fc_last_weight(last).detach().double()
    conv = interaction.conv_tp
    paths = []
    unique_products = set()
    for ins in conv.instructions:
        i1, i2, io = ins.i_in1, ins.i_in2, ins.i_out
        l1, l2, lo = conv.irreps_in1[i1].ir.l, conv.irreps_in2[i2].ir.l, conv.irreps_out[io].ir.l
        cg = o3.wigner_3j(l1,l2,lo)
        nnz = int((cg.abs() > 1e-14).sum())
        for m1,m2,mo in torch.nonzero(cg.abs()>1e-14).tolist():
            unique_products.add((l1,m1,l2,m2))
        paths.append({'input1_index': i1, 'input2_index': i2, 'output_index': io,
                      'l1': l1, 'l2': l2, 'lo': lo, 'CG_nonzero': nnz,
                      'output_components': 2*lo+1, 'mode': ins.connection_mode,
                      'path_weight': float(ins.path_weight)})
    channels = conv.irreps_in1[0].mul
    row = {'index': index, 'class': type(interaction).__name__,
           'linear_up_in': str(interaction.linear_up.irreps_in),
           'linear_up_out': str(interaction.linear_up.irreps_out),
           'conv_input1': str(conv.irreps_in1), 'conv_input2': str(conv.irreps_in2),
           'conv_output': str(conv.irreps_out), 'conv_weight_numel': conv.weight_numel,
           'linear_in': str(interaction.linear.irreps_in), 'linear_out': str(interaction.linear.irreps_out),
           'channels': channels, 'paths': paths, 'CG_nonzero_total': sum(p['CG_nonzero'] for p in paths),
           'unique_input_SH_component_products': len(unique_products),
           'within_output_sum_adds': sum(p['CG_nonzero']-p['output_components'] for p in paths),
           'output_components_per_channel': sum(p['output_components'] for p in paths),
           'radial_hidden_repr': str(hidden), 'radial_last_rank': rank_info(w),
           'radial_path_matrix_ranks': [rank_info(w[p*channels:(p+1)*channels]) for p in range(len(paths))]}
    if index == 1:
        p0 = next(i for i,p in enumerate(paths) if (p['l1'],p['l2'],p['lo']) == (1,1,0))
        p2 = next(i for i,p in enumerate(paths) if (p['l1'],p['l2'],p['lo']) == (1,1,2))
        pair_w = torch.stack([w[p0*channels],w[p2*channels]])
        row['Gaunt_two_output_path_row_rank_channel0'] = rank_info(pair_w)
        row['Gaunt_path_indices'] = [p0,p2]
        chain = radial._GenericChain(model,interaction,[7,21,37],torch.float64,'cpu')
        r = torch.linspace(1.,5.8,128,dtype=torch.float64)
        with torch.no_grad():
            a, _ = chain.with_derivative(r,0,0)
            weights = a[:,:w.shape[1]] @ w.T
        values = torch.stack([weights[:,p0*channels],weights[:,p2*channels]])
        row['Gaunt_sampled_radial_pair_rank_channel0'] = rank_info(values)
        row['Gaunt_sampled_points'] = {'pair_indices':[0,0], 'domain_A':[1.,5.8], 'points':128}
    result['interactions'].append(row)
for index, product in enumerate(model.products):
    result['products'].append({'index':index,'class':type(product).__name__,
                              'symmetric_contraction':str(product.symmetric_contractions),
                              'correlations': [getattr(c,'correlation',None) for c in product.symmetric_contractions.contractions],
                              'linear_in':str(product.linear.irreps_in),
                              'linear_out':str(product.linear.irreps_out)})
for index, readout in enumerate(model.readouts):
    activations = []
    for activation_name, activation in readout.named_modules():
        if hasattr(activation,'f'):
            f = activation.f
            activations.append({'name':activation_name,'wrapper':type(activation).__name__,
                                'function':getattr(f,'__name__',type(f).__name__)})
    result['readouts'].append({'index':index,'class':type(readout).__name__,'repr':str(readout),
                              'activations':activations})
for name, module in model.named_modules():
    if not (name.startswith('products.') or name.startswith('interactions.') or name.startswith('readouts.')):
        continue
    if not (hasattr(module,'irreps_in') and hasattr(module,'irreps_out') and hasattr(module,'weight')
            and hasattr(module,'instructions')):
        continue
    flat = module.weight.detach().flatten()
    offset = 0
    entry = {'in':str(module.irreps_in),'out':str(module.irreps_out),'matrices':[]}
    for instruction in module.instructions:
        shape = getattr(instruction,'path_shape',None)
        if shape is None:
            continue
        size = 1
        for dim in shape:
            size *= dim
        if len(shape) == 2 and offset + size <= flat.numel():
            entry['matrices'].append(rank_info(flat[offset:offset+size].reshape(shape)))
        offset += size
    if entry['matrices']:
        result['node_linear_matrices'][name] = entry
(OUT/'model.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'model':result['model_class'],'cutoff':result['r_max_A'], 'elements':len(result['atomic_numbers']),
    'parameters':result['parameters'],'interactions':[
        {k:v for k,v in row.items() if k not in ('radial_hidden_repr','paths','radial_path_matrix_ranks')}
        for row in result['interactions']]},indent=2))
