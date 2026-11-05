units metal
atom_style atomic
atom_modify map yes
newton on
boundary p p p
read_data data.lmp
pair_style symmetrix/mace/float32/kk no_domain_decomposition streamed_edges direct profile speed jit_device_artifact /tmp/sx-jit/artifacts/545f93799db86ee5331b1786360582c2cc23a5741cfc96ad29afa90f3ef85120/factorized_cuda_module_gen11.cubin jit_device_blocks_per_compute_unit 8
pair_coeff * * /data/symmetrix/mace-omat-0-medium.json O Sr Ti
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
timestep 0.001
fix integrate all nve
thermo_style custom step temp pe etotal
thermo 20
run 5
run 20 pre no
