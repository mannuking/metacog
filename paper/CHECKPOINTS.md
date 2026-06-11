# CHECKPOINTS

All Phase 1 LoRA checkpoints are stored in **Tinker's cloud storage** and referenced
by URI. To load a checkpoint for downstream eval, use the Tinker restore API.

## Phase 1 — Tinker URIs

| Step | URI | In-loop Acc | In-loop ECE | In-loop Conf |
|-----:|-----|------------:|------------:|-------------:|
|   5  | `tinker://4ee43737-7dd5-5271-98bd-fd6f28370006:train:0/weights/phase1_step5`  | 0.300 | 0.3525 | 0.647 |
|  10  | `tinker://4ee43737-7dd5-5271-98bd-fd6f28370006:train:0/weights/phase1_step10` | 0.550 | 0.2370 | 0.787 |
|  15  | `tinker://4ee43737-7dd5-5271-98bd-fd6f28370006:train:0/weights/phase1_step15` | 0.850 | 0.0980 | 0.948 |
|  20  | `tinker://4ee43737-7dd5-5271-98bd-fd6f28370006:train:0/weights/phase1_step20` | 0.800 | 0.0995 | 0.899 |

**Account:** `4ee43737-7dd5-5271-98bd-fd6f28370006`
**Training run:** `train:0`

## Loading a checkpoint

```python
from tinker import RESTClient, types

client = RESTClient()
ckpt = client.load_checkpoint(
    "tinker://4ee43737-7dd5-5271-98bd-fd6f28370006:train:0/weights/phase1_step20"
)
# ... use ckpt.model / ckpt.tokenizer as the new base
```

(Verify against the Tinker SDK docs — the API may differ slightly.)

## Notes

- These are **LoRA adapters only**, not full model checkpoints. The base model
  (`Qwen/Qwen3.6-35B-A3B`) is the standard public release and doesn't need to be
  re-uploaded.
- The Tinker account ID is the Tinker SDK account, not a personal one. Sharing the
  URIs publicly is safe (they're scoped to the Tinker account).
- For arXiv submission / permanent archival, the LoRA weights will be re-downloaded
  and stored in this repo under `checkpoints/phase1_step{N}/` (TODO Phase 2).
