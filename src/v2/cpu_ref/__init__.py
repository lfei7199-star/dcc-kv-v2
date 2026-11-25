"""CPU reference skeleton for DCC-KV Stage-2 (2A / 2C).

Scope: pure-Python, stdlib only. This is a *reference* implementation of the
merge math and the structural acceptance logic, NOT a GPU implementation.

Explicitly NOT implemented here (see README.md):
  - manual_decoder_forward
  - 84-round NCCL wiring
  - real all_to_all_v transport
  - any CUDA backend / sdp kernel
"""

BACKEND = "cpu_ref_f64"
DTYPE_BYTES = 8  # float64 on CPU reference; GPU arms must report bf16 = 2
