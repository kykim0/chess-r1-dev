"""Offline eval for verl-trained models.

Usage:
    python -m verl.trainer.main_offline_eval \
        --config-name offline_eval \
        actor_rollout_ref.model.path=/path/to/checkpoint \
        data.eval_files=/path/to/eval.parquet \
        trainer.n_gpus_per_node=4
"""

from collections import defaultdict
import json
import os
from pathlib import Path
from pprint import pprint
import socket
import uuid

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import ray
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_extern_type


@hydra.main(config_path="config", config_name="offline_eval", version_base=None)
def main(config: DictConfig):
    """Main entry point for offline evaluation with Hydra configuration management."""
    run_offline_eval(config)


def run_offline_eval(config: DictConfig) -> dict:
    """Runs offline evaluation of a model checkpoint.

    Args:
        config: Hydra configuration containing model path, data paths, and eval settings.

    Returns:
        Dict containing evaluation metrics.
    """
    # Initialize Ray if not already initialized.
    if not ray.is_initialized():
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        print(f"Initializing Ray with kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Create a remote instance of the EvalRunner.
    runner_cls = ray.remote(num_cpus=1)(EvalRunner)
    runner = runner_cls.remote()
    metrics = ray.get(runner.run.remote(config))

    return metrics


class EvalRunner:
    """Ray remote class for executing offline evaluation tasks."""

    def __init__(self):
        self.actor_rollout_wg = None

    def run(self, config: DictConfig) -> dict:
        """Executes the offline evaluation workflow.

        Args:
            config: Evaluation configuration.

        Returns:
            Dictionary of evaluation metrics.
        """
        print(f"EvalRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Setup model and tokenizer.
        local_model_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Load tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_model_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load reward manager for evaluation.
        reward_fn = load_reward_manager(config, tokenizer, is_train=False)

        # Create evaluation dataset and dataloader.
        eval_dataset = self._create_eval_dataset(config, tokenizer, processor)
        eval_dataloader = self._create_eval_dataloader(config, eval_dataset)

        # Initialize workers for generation.
        self._init_workers(config)

        # Run evaluation.
        metrics = self._evaluate(
            config=config,
            dataloader=eval_dataloader,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
        )

        # Save results.
        self._save_results(config, metrics)

        return metrics

    def _create_eval_dataset(
        self,
        config: DictConfig,
        tokenizer,
        processor,
    ) -> Dataset:
        """Creates evaluation dataset from parquet files.

        Args:
            config: Configuration containing data paths and settings.
            tokenizer: Tokenizer for text processing.
            processor: Optional processor for multimodal data.

        Returns:
            Evaluation dataset.
        """
        from verl.utils.dataset.rl_dataset import RLHFDataset

        eval_files = config.data.eval_files
        if not isinstance(eval_files, list):
            eval_files = [eval_files]

        # Check if custom dataset class is specified.
        if "custom_cls" in config.data and config.data.custom_cls.get("path", None):
            dataset_cls = load_extern_type(
                config.data.custom_cls.path,
                config.data.custom_cls.name
            )
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(
                    f"Custom dataset class must inherit from torch.utils.data.Dataset"
                )
        else:
            dataset_cls = RLHFDataset

        print(f"Using dataset class: {dataset_cls.__name__}")
        print(f"Loading evaluation data from: {eval_files}")

        dataset = dataset_cls(
            data_files=eval_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
            max_samples=config.data.get("max_samples", -1),
        )

        print(f"Loaded {len(dataset)} evaluation samples")
        return dataset

    def _create_eval_dataloader(
        self,
        config: DictConfig,
        dataset: Dataset,
    ) -> StatefulDataLoader:
        """Creates evaluation dataloader.

        Args:
            config: Configuration containing batch size and worker settings.
            dataset: Evaluation dataset.

        Returns:
            DataLoader for evaluation.
        """
        from verl.utils.dataset.rl_dataset import collate_fn

        batch_size = config.data.get("eval_batch_size", 32)
        num_workers = config.data.get("dataloader_num_workers", 0)
        shuffle = config.data.get("shuffle", False)

        dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=collate_fn,
        )

        print(f"Created dataloader with {len(dataloader)} batches (batch_size={batch_size})")
        return dataloader

    def _init_workers(self, config: DictConfig):
        """Initializes Ray workers for generation.

        Args:
            config: Configuration containing worker settings.
        """
        from verl.trainer.ppo.ray_trainer import Role

        # Determine the actor/rollout worker class based on strategy.
        strategy = config.actor_rollout_ref.get("actor", {}).get("strategy", "fsdp")
        if strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker
            actor_rollout_cls = ActorRolloutRefWorker
        elif strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker
            actor_rollout_cls = ActorRolloutRefWorker
        else:
            raise NotImplementedError(f"Strategy {strategy} not supported")

        # Create resource pool.
        n_gpus_per_node = config.trainer.n_gpus_per_node
        nnodes = config.trainer.nnodes
        resource_pool_spec = {
            "global_pool": [n_gpus_per_node] * nnodes,
        }

        resource_pool = RayResourcePool(
            process_on_nodes=resource_pool_spec["global_pool"],
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="global_pool",
        )

        # Create worker class with init args.
        actor_rollout_ray_cls = RayClassWithInitArgs(
            cls=ray.remote(actor_rollout_cls),
            config=config.actor_rollout_ref,
            role=str(Role.ActorRollout),
        )

        class_dict = {str(Role.ActorRollout): actor_rollout_ray_cls}
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)

        wg_kwargs = {
            "device_name": config.trainer.get("device", "cuda"),
        }

        wg_dict = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            **wg_kwargs,
        )

        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        self.actor_rollout_wg = spawn_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        print("Workers initialized successfully")

    def _evaluate(
        self,
        config: DictConfig,
        dataloader: StatefulDataLoader,
        tokenizer,
        reward_fn,
    ) -> dict:
        """Runs evaluation loop over all batches.

        Args:
            config: Evaluation configuration.
            dataloader: DataLoader yielding evaluation batches.
            tokenizer: Tokenizer for decoding outputs.
            reward_fn: Reward function for computing scores.

        Returns:
            Dictionary of evaluation metrics.
        """
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_uids = []

        n_samples = config.eval.get("n_samples", 1)

        print(f"\nStarting evaluation with {len(dataloader)} batches...")
        for _, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = DataProto.from_single_dict(batch_data)

            # Add unique IDs if not present.
            if "uid" not in batch.non_tensor_batch:
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object
                )

            # Repeat batch for multiple samples per prompt.
            if n_samples > 1:
                batch = batch.repeat(repeat_times=n_samples, interleave=True)

            # Store input texts.
            input_ids = batch.batch["input_ids"]
            input_texts = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)
            sample_uids.extend(batch.non_tensor_batch["uid"])

            # Store ground truths if available.
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in batch
            ]
            sample_gts.extend(ground_truths)

            # Prepare generation batch.
            gen_batch = self._get_gen_batch(batch)
            gen_batch.meta_info = {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": config.actor_rollout_ref.rollout.val_kwargs.get("do_sample", True),
                "validate": True,
            }

            # Pad to be divisible by dp_size.
            size_divisor = self.actor_rollout_wg.world_size
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)

            # Generate sequences.
            output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)

            # Unpad.
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

            # Decode generated outputs.
            output_ids = output_gen_batch.batch["responses"]
            output_texts = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            # Merge batch for reward computation.
            batch = batch.union(output_gen_batch)
            batch.meta_info["validate"] = True

            # Compute rewards.
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # Collect data sources.
            data_source_lst.append(
                batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        # Process all collected data.
        print(f"\nProcessed {len(sample_inputs)} total samples")

        # Compute aggregate metrics.
        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(
            data_sources, sample_uids, reward_extra_infos_dict
        )

        # Build final metrics dict.
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            for var_name, metric2val in var2metric2val.items():
                for metric_name, metric_val in metric2val.items():
                    pfx = f"eval/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Add overall statistics.
        metric_dict["eval/total_samples"] = len(sample_inputs)
        metric_dict["eval/mean_score"] = np.mean(sample_scores) if sample_scores else 0.0
        metric_dict["eval/std_score"] = np.std(sample_scores) if sample_scores else 0.0

        # Store generations for saving.
        self._generations = {
            "inputs": sample_inputs,
            "outputs": sample_outputs,
            "ground_truths": sample_gts,
            "scores": sample_scores,
            "uids": sample_uids,
            "reward_extra_info": dict(reward_extra_infos_dict),
        }

        # Print metrics.
        if config.eval.get("verbose", True):
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            for key, value in sorted(metric_dict.items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print("=" * 60)

        return metric_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Extracts generation batch from full batch.

        Args:
            batch: Full data batch.

        Returns:
            Batch prepared for generation.
        """
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        return gen_batch

    def _save_results(self, config: DictConfig, metrics: dict):
        """Saves evaluation results and generations to files.

        Args:
            config: Configuration containing output paths.
            metrics: Dictionary of evaluation metrics.
        """
        # Determine output directory.
        output_dir = config.eval.get("output_dir")
        if output_dir is None:
            output_dir = os.path.dirname(config.actor_rollout_ref.model.path)
            output_dir = os.path.join(output_dir, "eval_results")

        os.makedirs(output_dir, exist_ok=True)

        eval_files = config.data.eval_files
        if not isinstance(eval_files, list):
            eval_files = [eval_files]
        prefix = "_".join(Path(eval_file).stem for eval_file in eval_files)
        max_samples = config.data.get("max_samples", -1)
        if max_samples > 0:
            prefix += f"_n{max_samples}"

        # Save metrics.
        metrics_file = config.eval.get("output_file", "eval.json")
        metrics_path = os.path.join(output_dir, f"{prefix}_{metrics_file}")

        # Convert numpy types to Python types for JSON serialization.
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                serializable_metrics[k] = float(v)
            elif isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            else:
                serializable_metrics[k] = v

        with open(metrics_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")

        # Save generations if enabled
        if config.eval.get("save_generations", True) and hasattr(self, "_generations"):
            generations_file = config.eval.get("generations_file", "raw.jsonl")
            generations_path = os.path.join(output_dir, f"{prefix}_{generations_file}")

            with open(generations_path, "w") as f:
                n = len(self._generations["inputs"])
                for i in range(n):
                    entry = {
                        "input": self._generations["inputs"][i],
                        "output": self._generations["outputs"][i],
                        "ground_truth": self._generations["ground_truths"][i],
                        "score": self._generations["scores"][i],
                        "uid": str(self._generations["uids"][i]),
                    }
                    # Add extra reward info
                    for key, values in self._generations["reward_extra_info"].items():
                        if i < len(values):
                            entry[key] = values[i]
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"Generations saved to: {generations_path}")


if __name__ == "__main__":
    main()
