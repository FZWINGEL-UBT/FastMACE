"""CPU-only budgets for the outside-box follow-up; no timing claims."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path

out = Path(__file__).resolve().parent
repo = out.parents[2]
prior = json.loads((repo / 'plans/notes/omat0-ideas-round7/profile-arithmetic.json').read_text())
h = prior['H200_step_mean_us']
a = prior['A100_step_mean_us']
i2 = prior['same_lab_H200']['forward_sorted_ordered_us'] + prior['same_lab_H200']['wprime_real_us']
small = 2290.0  # Rounded measured whole-step value; see source below.
n, c, k, raw_dim, input_dim = 5000, 128, 32, 40, 4
r = {
    'source_commit': prior['research_source_commit'],
    'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
    'inputs': {
        'H200_step_us': h, 'A100_step_us': a,
        'H200_interaction2_lab_sum_us': i2,
        'H200_small_rounded_step_us': small,
        'warning': 'Step and kernel lab are different measurement contexts. No current A100 kernel split is inferred.',
        'sources': [
            'plans/notes/omat0-ideas-round7/profile-arithmetic.json',
            'results/slab-sorted-h200-20261108/h200-order/ab-order-ab.json',
            'results/slab-sorted-h200-20261108/h200-order/ab-order-ba.json',
            'results/slab-sorted-h200-20261108/h200-lab/spline_table_bound.json',
            'results/h200-omat0-small-20261108/README.md',
        ],
    },
    '20pct': {'H200_us': .2*h, 'A100_us': .2*a,
              'interaction2_replacement_budget_H200_us': i2-.2*h},
    'bottleneck_operator': {
        'channels_old': c, 'channels_student': k,
        'edge_work_ratio_assumption': k/c,
        'encoder_forward_backward_semantic_flops': 2*2*n*input_dim*c*k,
        'decoder_forward_backward_semantic_flops': 2*2*n*raw_dim*k*c,
        'old_decoder_forward_backward_semantic_flops': 2*2*n*raw_dim*c*c,
        'all_in_replacement_targets_us': [500, 800],
        'conditional_net_savings_us': [i2-800, i2-500],
        'conditional_step_fraction_saved': [(i2-800)/h, (i2-500)/h],
        'warning': 'Replacement targets are engineering goals, NOT measurements or predicted timings.',
    },
    'certified_skip': {
        'minimum_saved_cost_fraction_no_overhead': (.2*h)/i2,
        'minimum_saved_cost_fraction_overhead_100us': (.2*h+100)/i2,
        'minimum_saved_cost_fraction_overhead_200us': (.2*h+200)/i2,
        'hypothetical_skip_fractions': [.55, .70],
        'hypothetical_overhead_us': 150,
        'conditional_net_savings_us': [.55*i2-150, .70*i2-150],
        'warning': 'Fractions are cost-weighted work, not unweighted edge counts. No successful certificate or skip fraction has been observed.',
    },
    'exact_correction_rejection': {
        'original_radial_fmas_per_edge': 1280*64,
        'rank62_residual_fmas_per_edge': 62*(1280+64),
        'rank62_residual_over_original': 62*(1280+64)/(1280*64),
        'warning': '99 percent residual squared norm retained is not force-gate accuracy; exact residual requires rank64.',
    },
    'surrogate_HMC_separate_objective': {
        'teacher_endpoint_cost_budget_us': h,
        'teacher_endpoint_cost_status': 'Budget assumption based on whole-step time, not an energy-only measurement.',
        'proposal_32steps_us': 32*small+h,
        'cost_per_proposed_inner_step_us': small+h/32,
        'acceptance_needed_for_20pct_per_accepted_pathlength_proxy': (small+h/32)/(.8*h),
        'warning': 'This is NOT physical MD time or effective-sample-size speed. Reversibility, mixing and target-energy accuracy must be verified.',
    },
}
(out / 'budgets.json').write_text(json.dumps(r, indent=2)+'\n')
print(json.dumps(r, indent=2))
