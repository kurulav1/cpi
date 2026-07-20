#!/usr/bin/env python3
"""Reference oracle for the Qwen3.5 VISION tower.

Runs HuggingFace's Qwen3_5VisionModel on a DETERMINISTIC synthetic image and dumps the
activation after EVERY stage, so a Metal port can be bisected to a stage instead of being
compared only at the end. "The soft tokens are wrong" is a search across a patch embed, a
position interpolation, 12 blocks and a merger; a per-stage dump turns it into a lookup.

Only the vision weights are loaded (model.visual.*), not the text model.

The input is synthetic on purpose: no image file, no preprocessing pipeline, no PIL
dependency, and the same bytes on every machine. What is being gated here is the tower's
arithmetic, and an image decoder in the loop would be one more thing that can differ.

    python tools/qwen35_vision_oracle.py --model ~/models/qwen35-0.8b-hf \
                                         --out artifacts/q35_vision_oracle

Writes <out>/stage_NN_<name>.f32 (raw float32, C order) plus <out>/manifest.json holding
each stage's shape and the geometry the port has to reproduce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


def build_input(cfg: Qwen3_5VisionConfig, grid_h: int, grid_w: int):
    """A fixed, reproducible 'image', already patchified the way the tower expects.

    hidden_states is (num_patches, in_channels * temporal_patch_size * patch_size**2) --
    the tower's own patch_embed reshapes this into per-patch [C, T, P, P] volumes, so the
    flattening order here has to match that view exactly. Getting it wrong produces a
    tower that runs and is wrong, which is the failure mode this whole file exists to
    catch, so it is spelled out rather than inferred.
    """
    patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    n = grid_h * grid_w
    idx = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(patch_dim, dtype=torch.float32).unsqueeze(0)
    # Deterministic, non-trivial, bounded to [0,1], and varying along BOTH axes: a pattern
    # constant across patches would hide a patch-indexing bug, and one constant within a
    # patch would hide a channel-ordering bug.
    pixels = 0.5 + 0.5 * torch.sin(0.1 * idx + 0.037 * k)
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return pixels, grid_thw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HuggingFace model directory")
    ap.add_argument("--out", required=True, help="output directory for the dumps")
    ap.add_argument("--grid-h", type=int, default=8)
    ap.add_argument("--grid-w", type=int, default=8)
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg = Qwen3_5VisionConfig(**full_cfg["vision_config"])

    # The grid must be a whole number of merge units in both axes -- the merger folds
    # spatial_merge_size**2 patches into one soft token and a partial unit has nothing to
    # fold. Caught here rather than as a reshape error 200 lines deeper.
    if args.grid_h % cfg.spatial_merge_size or args.grid_w % cfg.spatial_merge_size:
        raise SystemExit(
            "grid must be a multiple of spatial_merge_size=%d in both axes" % cfg.spatial_merge_size
        )

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit("no .safetensors in %s" % model_dir)
    prefix = "model.visual."
    visual = {}
    for shard in shards:
        for name, tensor in load_file(str(shard)).items():
            if name.startswith(prefix):
                visual[name[len(prefix):]] = tensor.to(torch.float32)
    if not visual:
        raise SystemExit("no %s* tensors found -- is this a multimodal checkpoint?" % prefix)

    torch.set_grad_enabled(False)
    model = Qwen3_5VisionModel(cfg).to(torch.float32).eval()
    missing, unexpected = model.load_state_dict(visual, strict=False)
    # strict=False is needed because buffers (rotary inv_freq) are not stored in the
    # checkpoint, but a missing WEIGHT would silently leave random init in the tower and
    # every comparison after that would be meaningless. So: allow missing buffers, refuse
    # missing parameters.
    param_names = {n for n, _ in model.named_parameters()}
    missing_params = [n for n in missing if n in param_names]
    if missing_params:
        raise SystemExit("checkpoint is missing %d vision PARAMETERS, e.g. %s"
                         % (len(missing_params), missing_params[:5]))
    if unexpected:
        print("[warn] %d unexpected tensors ignored, e.g. %s" % (len(unexpected), unexpected[:3]))

    pixels, grid_thw = build_input(cfg, args.grid_h, args.grid_w)

    stages: list[tuple[str, torch.Tensor]] = []

    def record(name):
        def hook(_mod, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            stages.append((name, t.detach().to(torch.float32).contiguous()))
        return hook

    handles = [model.patch_embed.register_forward_hook(record("patch_embed"))]
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(record("block_%02d" % i)))
    handles.append(model.merger.register_forward_hook(record("merger")))

    out = model(pixels, grid_thw)
    for h in handles:
        h.remove()

    # The tower adds interpolated position embeddings between patch_embed and block 0, and
    # that add is not a module, so no hook sees it. Recomputing it here is what makes the
    # patch_embed -> block_00 step bisectable rather than a two-stage jump.
    pos = model.fast_pos_embed_interpolate(grid_thw).detach().to(torch.float32).contiguous()
    stages.insert(1, ("pos_embed", pos))
    stages.insert(2, ("pos_embed_added", (stages[0][1] + pos).contiguous()))

    manifest = {
        "source": str(model_dir),
        "grid_thw": grid_thw.tolist(),
        "geometry": {
            "grid_t": int(grid_thw[0][0]),
            "grid_h": int(grid_thw[0][1]),
            "grid_w": int(grid_thw[0][2]),
            "num_grid_per_side": int(cfg.num_position_embeddings ** 0.5),
            "depth": cfg.depth,
            "hidden_size": cfg.hidden_size,
            "num_heads": cfg.num_heads,
            "head_dim": cfg.hidden_size // cfg.num_heads,
            "intermediate_size": cfg.intermediate_size,
            "patch_size": cfg.patch_size,
            "temporal_patch_size": cfg.temporal_patch_size,
            "in_channels": cfg.in_channels,
            "spatial_merge_size": cfg.spatial_merge_size,
            "num_position_embeddings": cfg.num_position_embeddings,
            "out_hidden_size": cfg.out_hidden_size,
            "hidden_act": cfg.hidden_act,
            "layernorm_eps": 1e-6,
        },
        "stages": [],
    }

    def write(name: str, t: torch.Tensor, index: int) -> None:
        path = out_dir / ("stage_%02d_%s.f32" % (index, name))
        path.write_bytes(t.numpy().astype("<f4").tobytes())
        manifest["stages"].append({"index": index, "name": name, "shape": list(t.shape),
                                   "file": path.name})
        print("  %-18s %-18s %s" % (name, tuple(t.shape), path.name))

    print("[oracle] grid=%dx%d patches=%d" % (args.grid_h, args.grid_w, args.grid_h * args.grid_w))
    write("input_pixels", pixels, 0)
    for i, (name, t) in enumerate(stages, start=1):
        write(name, t, i)
    # NOT the tower's product. last_hidden_state is the pre-merge block output (num_patches,
    # hidden_size); the soft tokens the text model actually consumes are the MERGER's output
    # (num_patches / spatial_merge_size**2, out_hidden_size) -- stage_16 above. Dumped anyway
    # because it is a free extra checkpoint, but named so it cannot be mistaken for the result.
    out_t = out if isinstance(out, torch.Tensor) else out.last_hidden_state
    write("last_hidden_state_premerge", out_t.detach().to(torch.float32).contiguous(),
          len(stages) + 1)
    manifest["soft_tokens_stage"] = "merger"

    # The vision weights, in the same fp32 layout, so a port can be gated before the container
    # format carries them. The converter does not emit model.visual.* yet, and waiting for it
    # would mean the arithmetic is only checkable once the plumbing is done -- which is the
    # wrong order. These stay useful afterwards as the test's fixture.
    wdir = out_dir / "weights"
    wdir.mkdir(exist_ok=True)
    manifest["weights"] = []
    for name, tensor in sorted(visual.items()):
        fname = name.replace(".", "_") + ".f32"
        (wdir / fname).write_bytes(tensor.contiguous().numpy().astype("<f4").tobytes())
        manifest["weights"].append({"name": name, "shape": list(tensor.shape),
                                    "file": "weights/" + fname})
    print("[oracle] wrote %d weight tensors" % len(manifest["weights"]))

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("[oracle] wrote %d stages to %s" % (len(manifest["stages"]), out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
