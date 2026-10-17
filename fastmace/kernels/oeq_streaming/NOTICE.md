# Vendored code

`factorized_projected.cuh` and `schedule.py` (both adapted) come from
OpenEquivariance pull request 223, "Add receiver-streaming UVU convolution
kernel" by Paul Fuchs (https://github.com/PASSIONLab/OpenEquivariance/pull/223,
commit ef051c3f9efa2d1daf74cfbe3beaad152235f47d, not merged as of
14 October 2026), which adapts the streaming schedule of Chorošajev and Bény,
"Sobek: Streaming Equivariant Tensor Product Convolutions",
arXiv:2607.18074. OpenEquivariance is distributed under the BSD 3-Clause
licence; its copyright notice and licence terms apply to these files.
`jinja_env.py` adapts the PR's `templates/jinja_utils.py`.

Changes for this project: the template's node-feature offsets go through one
macro that also supports the `ir_mul` layout (component-major,
channel-minor), the schedule's accumulator slots keep their segment start
and component, and the schedule passes its layout to the template.
