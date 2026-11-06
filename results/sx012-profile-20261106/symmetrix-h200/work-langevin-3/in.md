units metal
atom_style atomic
atom_modify map yes
newton on
boundary p p p
read_data data.lmp
pair_style symmetrix/mace/float32/kk no_domain_decomposition streamed_edges direct profile speed jit_device_artifact /root/sx-jit/artifacts/9d3cc336c46d7566ba83cff5b5472aee9bb5354eb7d946b270d15b453554cb5f/factorized_cuda_module_gen11_src33ddd9e22b6c34167906f516d0c4857aabe8f131d4027ff75e1cb3d4277b8890.cubin jit_device_blocks_per_compute_unit 8
pair_coeff * * /data/symmetrix-0.1.2/mace-omat-0-medium.json O Sr Ti
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
timestep 0.001
fix integrate all nve
fix thermostat all langevin 300.0 300.0 0.500000 1 zero yes
thermo_style custom step temp pe etotal
thermo 20
run 5
run 20 pre no
