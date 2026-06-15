"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,bpb,sample

Examples:

    # Evaluate a HuggingFace model (e.g. GPT-2 124M) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse
import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model, find_largest_model, find_last_step
from nanochat.core_eval import evaluate_task
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# HuggingFace loading utilities

class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanochat-compatible interface."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes

# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_core(model, tokenizer, device, max_per_task=-1, seed=1337):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, and core_metric.

    ``seed`` controls both the (deterministic) subsample/shuffle of each task's data and
    the few-shot example selection, so that multi-seed evaluation produces independent
    draws while each individual seed remains fully reproducible.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' '),
            'fewshot_seed': 1234 + seed,
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling when using max_per_task
        shuffle_rng = random.Random(seed)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "seed": seed,
    }
    return out

# -----------------------------------------------------------------------------
# Seeding / reproducibility

def set_seed(seed: int):
    """Explicitly seed all RNGs so that an eval run is reproducible across invocations."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summarize(values):
    """Return (mean, std, n) of a list of floats; std is the population std (0 for n<2)."""
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5, n


# -----------------------------------------------------------------------------
# Idempotency: per-checkpoint eval results live in <checkpoint_dir>/evaluation/eval_<step>.json

# Eval modes that persist a result into the eval JSON ('sample' is stdout-only).
PERSISTENT_EVAL_MODES = {"core", "bpb"}


def _eval_json_path(checkpoint_dir, step):
    return os.path.join(checkpoint_dir, "evaluation", f"eval_{step:06d}.json")


def _load_existing_eval(checkpoint_dir, step):
    """Return the parsed per-checkpoint eval record, or None if absent/unreadable."""
    path = _eval_json_path(checkpoint_dir, step)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _eval_already_complete(checkpoint_dir, step, eval_modes, seeds):
    """True iff the saved eval record already covers every requested persistent mode
    (core/bpb) and, for CORE, every requested seed. 'sample' writes no artifact, so it
    never counts toward (nor blocks) completeness."""
    data = _load_existing_eval(checkpoint_dir, step)
    if data is None:
        return False
    required = {m for m in eval_modes if m in PERSISTENT_EVAL_MODES}
    if not required:
        return False  # e.g. a sample-only request: nothing persistent to reuse
    if not all(m in data for m in required):
        return False
    if "core" in required:
        done_seeds = {str(s) for s in data.get("seeds", [])}
        if not all(str(s) in done_seeds for s in seeds):
            return False
    return True


# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--seeds', type=str, default='1337', help='Comma-separated eval seeds. Multiple seeds run CORE repeatedly and report mean +/- std.')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    parser.add_argument('--force', action='store_true', help='Re-run evaluation even if results already exist for this checkpoint.')
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Parse eval seeds (one or more). Multiple seeds give CORE mean +/- std.
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    if not seeds:
        parser.error("No valid --seeds provided")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # Load model and tokenizer
    is_hf_model = args.hf_path is not None
    checkpoint_dir = None
    resolved_step = None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_tag = args.hf_path.replace("/", "-")
        model_slug = model_tag
    else:
        # Resolve the checkpoint (model_tag + step) WITHOUT loading weights first, so an
        # already-evaluated checkpoint can be skipped before the expensive model build and
        # eval-bundle download. Auto-guess the largest model tag when none is given; this
        # also catches the case where two variants collide on a single model_tag.
        base_dir = get_base_dir()
        checkpoints_root = os.path.join(base_dir, "base_checkpoints")
        model_tag = args.model_tag if args.model_tag else find_largest_model(checkpoints_root)
        checkpoint_dir = os.path.join(checkpoints_root, model_tag)
        resolved_step = args.step if args.step is not None else find_last_step(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_{resolved_step:06d}.pt")
        print0(f"Resolved checkpoint: model_tag={model_tag} | step={resolved_step} | path={checkpoint_path}")

        # Idempotency: if every requested persistent eval (core/bpb) already exists for this
        # exact checkpoint + seeds, do nothing. All ranks read the same JSON and decide alike.
        if not args.force and _eval_already_complete(checkpoint_dir, resolved_step, eval_modes, seeds):
            print0(f"Eval results already complete for model_tag={model_tag} step={resolved_step} "
                   f"(modes={sorted(eval_modes)}, seeds={seeds}); skipping. Use --force to re-run.")
            compute_cleanup()
            return

        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=resolved_step)
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(device=device)
        resolved_step = meta['step']
        model_name = f"base_model (step {resolved_step})"
        # Key all CORE artifacts by model_tag AND step so that distinct variants that finish
        # at the same step never overwrite each other's results (the prior step-only slug
        # `base_model_<step>` was the root cause of identical CORE across variants).
        model_slug = f"{model_tag}_{resolved_step:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")
    print0(f"Eval seeds: {seeds}")

    # Results to log
    core_results = None
    bpb_results = {}
    loss_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if 'sample' in eval_modes and not is_hf_model:
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
        if ddp_rank == 0:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)
    elif 'sample' in eval_modes and is_hf_model:
        print0("\nSkipping sampling for HuggingFace models (not supported)")

    # --- BPB evaluation (continuous primary metric: bpb + raw cross-entropy loss) ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        # Seed before BPB so the (deterministic) eval data ordering is reproducible.
        set_seed(seeds[0])
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
            stats = evaluate_bpb(model, loader, steps, token_bytes, return_stats=True)
            bpb_results[split_name] = stats["bpb"]
            loss_results[split_name] = stats["loss"]
            print0(f"{split_name} bpb: {stats['bpb']:.6f} | {split_name} loss: {stats['loss']:.6f}")

    # --- CORE evaluation (possibly across multiple seeds for mean +/- std) ---
    core_per_seed = []
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        for seed in seeds:
            print0(f"\n--- CORE seed {seed} ---")
            set_seed(seed)
            core_per_seed.append(evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task, seed=seed))

        # The reported core_results is the last seed's full breakdown; the aggregate
        # mean/std across seeds is what callers should treat as the headline number.
        core_results = core_per_seed[-1]
        core_metrics = [r["core_metric"] for r in core_per_seed]
        core_mean, core_std, _ = _summarize(core_metrics)
        if len(seeds) > 1:
            print0(f"CORE metric: mean={core_mean:.4f} std={core_std:.4f} over seeds {seeds}")
        else:
            print0(f"CORE metric: {core_mean:.4f}")

        # Write per-(model_tag, step) CSV — keyed by model_slug so distinct variants that
        # finish at the same step never overwrite one another.
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_mean:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")

    # --- Write canonical per-checkpoint evaluation JSON ---
    # results.py / run_evaluation.py read this file, keyed by (model_tag, step). Writing it
    # here is what stops the step-only CSV fallback (the source of identical CORE across
    # variants) from being load-bearing. HF models have no checkpoint dir, so skip.
    # Merge into any existing record so modes evaluated in separate runs accumulate (e.g. a
    # bpb-only run followed by a core-only run keeps both) and the idempotency guard above
    # sees the union of completed modes.
    if ddp_rank == 0 and checkpoint_dir is not None:
        eval_dir = os.path.join(checkpoint_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        eval_json_path = _eval_json_path(checkpoint_dir, resolved_step)
        eval_record = _load_existing_eval(checkpoint_dir, resolved_step) or {}
        eval_record["model_tag"] = model_tag
        eval_record["step"] = resolved_step
        eval_record["checkpoint_path"] = os.path.join(checkpoint_dir, f"model_{resolved_step:06d}.pt")
        eval_record["eval_modes"] = sorted(set(eval_record.get("eval_modes", [])) | eval_modes)
        if 'bpb' in eval_modes:
            eval_record["bpb"] = bpb_results
            eval_record["val_bpb"] = bpb_results.get("val")
            eval_record["loss"] = loss_results
            eval_record["val_loss"] = loss_results.get("val")
        if core_per_seed:
            core_metrics = [r["core_metric"] for r in core_per_seed]
            core_mean, core_std, n_seeds = _summarize(core_metrics)
            eval_record["core"] = {
                "core_metric": core_mean,        # headline = mean across seeds
                "core_metric_mean": core_mean,
                "core_metric_std": core_std,
                "num_seeds": n_seeds,
                "per_seed": {str(r["seed"]): r["core_metric"] for r in core_per_seed},
                "centered_results": core_results["centered_results"],
            }
            eval_record["seeds"] = seeds          # seeds backing the CORE result
        eval_record.setdefault("seeds", seeds)
        with open(eval_json_path, 'w', encoding='utf-8') as f:
            json.dump(eval_record, f, indent=2)
        print0(f"Canonical eval JSON written to: {eval_json_path}")

    # --- Log to report ---
    from nanochat.report import get_report
    report_data = [{"model": model_name}]

    if core_per_seed:
        core_mean, core_std, n_seeds = _summarize([r["core_metric"] for r in core_per_seed])
        report_data[0]["CORE metric"] = core_mean
        if n_seeds > 1:
            report_data[0]["CORE metric std"] = core_std
            report_data[0]["CORE seeds"] = seeds
        report_data.append(core_results["centered_results"])

    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get("train")
        report_data[0]["val bpb"] = bpb_results.get("val")
        report_data[0]["train loss"] = loss_results.get("train")
        report_data[0]["val loss"] = loss_results.get("val")

    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)})

    get_report().log(section="Base model evaluation", data=report_data)

    compute_cleanup()


if __name__ == "__main__":
    main()
