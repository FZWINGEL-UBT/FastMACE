units metal
atom_style atomic
atom_modify map yes
newton on
boundary p p p
read_data data.lmp
pair_style symmetrix/mace/float32/kk no_domain_decomposition streamed_edges direct profile speed jit_device_artifact /home/fzwin/tmp/symmetrix-rematch/cache/jit/artifacts/49f9abffcdbd4b7eccdb1a24dce2407b24c89af3c211b7c706a2bd13bb3e5700/factorized_cuda_module_gen11.cubin jit_device_blocks_per_compute_unit 8
pair_coeff * * /home/fzwin/tmp/symmetrix-rematch/models/mace-omat-0-medium.json Cl Na
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
timestep 0.001
fix integrate all nve
fix thermostat all langevin 300.0 300.0 0.500000 20261105 zero yes
thermo_style custom step temp pe etotal
thermo 3000
run 500
run 3000 pre no
