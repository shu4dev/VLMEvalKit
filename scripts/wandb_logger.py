#!/usr/bin/env python3
"""
WandB Logger for VLMEvalKit
Log VLMEvalKit evaluation results to Weights & Biases for experiment tracking.

Usage:
    # Log results from a specific run (single dataset)
    python scripts/wandb_logger.py --model GPT4o --dataset MMBench_DEV_EN --work-dir ./outputs

    # Log results from multiple datasets (like run.py)
    python scripts/wandb_logger.py --model GPT4o --data MMBench_DEV_EN MMBench_DEV_CN MMMU_DEV_VAL --work-dir ./outputs

    # Log all existing results in work directory
    python scripts/wandb_logger.py --log-all --work-dir ./outputs

    # Run evaluation and log to WandB in one command (single dataset)
    python scripts/wandb_logger.py --run-and-log --model GPT4o --dataset MMBench_DEV_EN

    # Run evaluation for multiple datasets and log to WandB
    python scripts/wandb_logger.py --run-and-log --model GPT4o --data MMBench_DEV_EN MMBench_DEV_CN MMMU_DEV_VAL

    # Run evaluation with VLLM batch processing and log results (multiple datasets)
    python scripts/wandb_logger.py --run-and-log --model molmo-7B-D-0924 --data MMBench_DEV_EN MMMU_DEV_VAL --use-vllm --batch-size 4

    # Run with verbose batch processing monitoring
    python scripts/wandb_logger.py --run-and-log --model molmo-7B-D-0924 --data MMBench_DEV_EN --use-vllm --batch-size 4 --verbose

    # Run evaluation
    python scripts/wandb_logger.py --run-and-log --model GPT4o --data MMBench_DEV_EN

    # Run with custom max output tokens override
    python scripts/wandb_logger.py --run-and-log --model GPT4o --data MMBench_DEV_EN --max-output-tokens 2048

    # Run with custom model detection
    python scripts/wandb_logger.py --run-and-log --pass-custom-model Qwen/Qwen2-VL-7B-Instruct --data MMBench_DEV_EN

    # Combine custom model with existing models  
    python scripts/wandb_logger.py --run-and-log --model GPT4o --pass-custom-model microsoft/Phi-3-vision-128k-instruct --data MMBench_DEV_EN
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Add parent directory to path to import vlmeval modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    import wandb
except ImportError:
    print("ERROR: wandb is not installed. Please install it with: pip install wandb")
    sys.exit(1)

from vlmeval.smp import load, get_logger
from vlmeval.config import supported_VLM
from vlmeval.dataset import SUPPORTED_DATASETS


logger = get_logger('WandB Logger')


def extract_metrics_from_result_file(result_file: str) -> Dict[str, Any]:
    """Extract metrics from VLMEvalKit result files."""
    metrics = {}
    
    try:
        if result_file.endswith('.xlsx'):
            df = pd.read_excel(result_file)
        elif result_file.endswith('.csv'):
            df = pd.read_csv(result_file)
        elif result_file.endswith('.json'):
            with open(result_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            else:
                metrics['raw_results'] = data
                return metrics
        elif result_file.endswith('.pkl'):
            data = load(result_file)
            if isinstance(data, dict):
                return data
            else:
                metrics['raw_results'] = str(data)
                return metrics
        else:
            logger.warning(f"Unsupported file format: {result_file}")
            return metrics
            
        # Extract common metrics from DataFrame
        if 'prediction' in df.columns and 'answer' in df.columns:
            # Calculate accuracy if we have predictions and ground truth
            correct = (df['prediction'] == df['answer']).sum()
            total = len(df)
            metrics['accuracy'] = correct / total if total > 0 else 0.0
            metrics['correct_predictions'] = int(correct)
            metrics['total_samples'] = int(total)
            
        # Extract any numerical columns as metrics
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col not in ['index', 'line_id']:  # Skip identifier columns
                metrics[f'{col}_mean'] = float(df[col].mean())
                metrics[f'{col}_std'] = float(df[col].std())
                
        # Add dataset statistics
        metrics['dataset_size'] = len(df)
        
        # Extract unique categories if present
        if 'category' in df.columns:
            categories = df['category'].value_counts().to_dict()
            metrics['categories'] = {str(k): int(v) for k, v in categories.items()}
            
    except Exception as e:
        logger.error(f"Error extracting metrics from {result_file}: {e}")
        metrics['extraction_error'] = str(e)
        
    return metrics


def find_evaluation_files(work_dir: str, model_name: str, dataset_name: str) -> List[str]:
    """Find all evaluation result files for a model-dataset combination."""
    model_dir = Path(work_dir) / model_name
    if not model_dir.exists():
        return []
        
    result_files = []
    patterns = [
        f"{model_name}_{dataset_name}.xlsx",
        f"{model_name}_{dataset_name}.csv", 
        f"{model_name}_{dataset_name}.json",
        f"{model_name}_{dataset_name}_*.xlsx",
        f"{model_name}_{dataset_name}_*.csv",
        f"{model_name}_{dataset_name}_*.json",
    ]
    
    for pattern in patterns:
        result_files.extend(model_dir.glob(pattern))
        
    return [str(f) for f in result_files]


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Extract model configuration if available."""
    config = {}
    
    if model_name in supported_VLM:
        model_partial = supported_VLM[model_name]
        if hasattr(model_partial, 'keywords'):
            config.update(model_partial.keywords)
        if hasattr(model_partial, 'func'):
            config['model_class'] = model_partial.func.__name__
            
    return config


def log_to_wandb(
    model_name: str, 
    dataset_name: str, 
    result_files: List[str],
    project: str = "vlmeval-benchmark",
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    use_vllm: bool = False,
    batch_size: Optional[int] = None
) -> str:
    """Log evaluation results to WandB."""
    
    # Prepare WandB configuration
    config = {
        "model": model_name,
        "dataset": dataset_name,
        "framework": "VLMEvalKit",
        "result_files": [os.path.basename(f) for f in result_files],
        "use_vllm": use_vllm,
        "batch_size": batch_size
    }
    
    # Add model-specific configuration
    model_config = get_model_config(model_name)
    if model_config:
        config["model_config"] = model_config
        
    # Add dataset info if available
    if dataset_name in SUPPORTED_DATASETS:
        config["dataset_supported"] = True
    
    # Initialize WandB run
    run = wandb.init(
        project=project,
        name=f"{model_name}_{dataset_name}",
        config=config,
        tags=tags or [model_name, dataset_name],
        notes=notes
    )
    
    # Log metrics from each result file
    all_metrics = {}
    for result_file in result_files:
        file_metrics = extract_metrics_from_result_file(result_file)
        file_suffix = Path(result_file).stem.replace(f"{model_name}_{dataset_name}", "").lstrip("_")
        
        if file_suffix:
            # Prefix metrics with file suffix if multiple files
            prefixed_metrics = {f"{file_suffix}_{k}": v for k, v in file_metrics.items()}
        else:
            prefixed_metrics = file_metrics
            
        all_metrics.update(prefixed_metrics)
        
        # Upload result file as artifact
        artifact = wandb.Artifact(
            name=f"results_{os.path.basename(result_file)}", 
            type="evaluation_results"
        )
        artifact.add_file(result_file)
        run.log_artifact(artifact)
    
    # Log all metrics
    wandb.log(all_metrics)
    
    run_url = run.url
    wandb.finish()
    
    return run_url


def run_evaluation_and_log(
    model_name: str,
    dataset_names: List[str], 
    work_dir: str = "./outputs",
    project: str = "vlmeval-benchmark",
    use_vllm: bool = False,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    max_output_tokens: Optional[int] = None,
    pass_custom_model: Optional[str] = None,
    additional_args: List[str] = None
) -> List[str]:
    """Run VLMEvalKit evaluation and log results to WandB."""
    
    logger.info(f"Running evaluation for {model_name} on {len(dataset_names)} datasets: {', '.join(dataset_names)}")
    
    # Log batch processing configuration
    if use_vllm:
        if batch_size is not None:
            logger.info(f"Using VLLM with batch processing: batch_size={batch_size}")
        else:
            logger.info("Using VLLM with sequential processing")
    else:
        logger.info("Using transformers backend")
    
    # Prepare run command
    cmd = [sys.executable, "run.py"]
    
    # Add model if specified (can be used together with custom model)
    if model_name:
        cmd.extend(["--model", model_name])
    
    # Add datasets
    cmd.extend(["--data"])
    cmd.extend(dataset_names)
    
    # Add work directory
    cmd.extend(["--work-dir", work_dir])
    
    # Add VLLM and batch processing arguments
    if use_vllm:
        cmd.append("--use-vllm")
        
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
        
    if verbose:
        cmd.append("--verbose")
    
    
    # Add max output tokens override
    if max_output_tokens is not None:
        cmd.extend(["--max-output-tokens", str(max_output_tokens)])
    
    # Add custom model support
    if pass_custom_model is not None:
        cmd.extend(["--pass-custom-model", pass_custom_model])
    
    if additional_args:
        cmd.extend(additional_args)
        
    # Run evaluation with real-time output
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True, 
            bufsize=1,  # Line buffered
            universal_newlines=True,
            cwd=Path(__file__).parent.parent
        )
        
        # Stream output in real-time
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Print to both stdout and stderr with explicit flushing for SLURM
                output_line = output.strip()
                print(output_line)
                sys.stdout.flush()  # Force flush to SLURM log
                print(output_line, file=sys.stderr)
                sys.stderr.flush()  # Force flush to SLURM error log
                output_lines.append(output_line)
        
        # Wait for process to complete
        return_code = process.poll()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, output='\n'.join(output_lines))
            
        logger.info("Evaluation completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed with return code {e.returncode}")
        if hasattr(e, 'output') and e.output:
            logger.error(f"Command output: {e.output}")
        raise
    
    # Find and log result files for each dataset
    run_urls = []
    for dataset_name in dataset_names:
        result_files = find_evaluation_files(work_dir, model_name, dataset_name)
        if not result_files:
            logger.warning(f"No result files found for {model_name}_{dataset_name}")
            continue
            
        logger.info(f"Found {len(result_files)} result files for {dataset_name}: {result_files}")
        
        # Log to WandB
        run_url = log_to_wandb(
            model_name=model_name,
            dataset_name=dataset_name, 
            result_files=result_files,
            project=project,
            use_vllm=use_vllm,
            batch_size=batch_size,
            notes=f"Automated run via wandb_logger.py"
        )
        
        logger.info(f"Results for {dataset_name} logged to WandB: {run_url}")
        run_urls.append(run_url)
    
    return run_urls


def log_all_existing_results(work_dir: str, project: str = "vlmeval-benchmark"):
    """Log all existing evaluation results in work directory to WandB."""
    
    work_path = Path(work_dir)
    if not work_path.exists():
        logger.error(f"Work directory does not exist: {work_dir}")
        return
        
    logged_count = 0
    
    for model_dir in work_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        # Find all result files in model directory
        result_files = list(model_dir.glob("*.xlsx")) + list(model_dir.glob("*.csv")) + list(model_dir.glob("*.json"))
        
        # Group by dataset
        datasets = set()
        for result_file in result_files:
            filename = result_file.stem
            if filename.startswith(f"{model_name}_"):
                dataset_part = filename[len(f"{model_name}_"):]
                # Remove suffix like _acc, _score etc
                dataset_name = dataset_part.split('_')[0]
                datasets.add(dataset_name)
                
        for dataset_name in datasets:
            dataset_files = find_evaluation_files(work_dir, model_name, dataset_name)
            if dataset_files:
                logger.info(f"Logging {model_name} x {dataset_name}")
                try:
                    run_url = log_to_wandb(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        result_files=dataset_files, 
                        project=project,
                        notes="Batch upload of existing results"
                    )
                    logger.info(f"Logged: {run_url}")
                    logged_count += 1
                except Exception as e:
                    logger.error(f"Failed to log {model_name} x {dataset_name}: {e}")
                    
    logger.info(f"Successfully logged {logged_count} evaluations to WandB")


def main():
    parser = argparse.ArgumentParser(description="Log VLMEvalKit results to WandB")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--dataset", type=str, help="Dataset name (for single dataset compatibility)")
    parser.add_argument("--data", type=str, nargs="+", help="Dataset names (supports multiple datasets like run.py)")
    parser.add_argument("--work-dir", type=str, default="./outputs", help="Work directory")
    parser.add_argument("--project", type=str, default="vlmeval-benchmark", help="WandB project name")
    parser.add_argument("--run-and-log", action="store_true", help="Run evaluation and log results")
    parser.add_argument("--log-all", action="store_true", help="Log all existing results")
    parser.add_argument("--tags", type=str, nargs="+", help="Additional tags for WandB run")
    parser.add_argument("--notes", type=str, help="Notes for WandB run")
    
    # VLLM and batch processing arguments
    parser.add_argument("--use-vllm", action="store_true", help="Use VLLM for inference")
    parser.add_argument("--batch-size", type=int, help="Batch size for VLLM inference (requires --use-vllm)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    
    # Global token override
    parser.add_argument(
        "--max-output-tokens", type=int, default=None,
        help="Global override for maximum output tokens. Supersedes all model-specific and dataset-specific token limits."
    )
    
    # Custom model support
    parser.add_argument(
        "--pass-custom-model", type=str, default=None,
        help="Path to a HuggingFace repository for automatic model detection and evaluation. "
             "The system will automatically detect the model architecture and use appropriate default settings."
    )
    
    # Pass through additional arguments to run.py
    parser.add_argument("--run-args", type=str, nargs=argparse.REMAINDER, help="Additional arguments for run.py")
    
    args = parser.parse_args()
    
    # Handle dataset arguments - prioritize --data over --dataset for consistency with run.py
    datasets = []
    if args.data:
        datasets = args.data
    elif args.dataset:
        datasets = [args.dataset]
    
    # Validate batch processing arguments
    if args.batch_size is not None and not args.use_vllm:
        logger.warning("--batch-size specified without --use-vllm. Batch processing requires VLLM backend.")
        logger.info("Adding --use-vllm automatically.")
        args.use_vllm = True
    
    if args.batch_size is not None and args.batch_size <= 1:
        logger.warning(f"Invalid batch size {args.batch_size}. Batch size must be > 1. Disabling batch processing.")
        args.batch_size = None
    
    # Initialize WandB if not already done
    if not wandb.api.api_key:
        logger.info("WandB not configured. Please run 'wandb login' first.")
        return
        
    if args.log_all:
        log_all_existing_results(args.work_dir, args.project)
        
    elif args.run_and_log:
        if not args.model and not args.pass_custom_model:
            logger.error("Either --model or --pass-custom-model is required for --run-and-log")
            return
        if not datasets:
            logger.error("Datasets (--data or --dataset) are required for --run-and-log")
            return
            
        # Determine model name for WandB logging and run.py
        # If both model and custom model are specified, pass the regular model name to run.py
        # If only custom model is specified, use the registered model name
        if args.model:
            wandb_model_name = args.model
        elif args.pass_custom_model:
            # Mirror register_custom_model naming so run.py sees the same alias
            if args.pass_custom_model.startswith('/LOCAL_MODEL'):
                local_model_path = args.pass_custom_model[len('/LOCAL_MODEL'):]
                wandb_model_name = (
                    local_model_path.replace('/', '_')
                    if local_model_path else 'LOCAL_MODEL'
                )
            else:
                wandb_model_name = (
                    args.pass_custom_model.replace('/', '_')
                    if '/' in args.pass_custom_model else args.pass_custom_model
                )
        else:
            wandb_model_name = None
        
        run_urls = run_evaluation_and_log(
            model_name=wandb_model_name,
            dataset_names=datasets,
            work_dir=args.work_dir,
            project=args.project,
            use_vllm=args.use_vllm,
            batch_size=args.batch_size,
            verbose=args.verbose,
            max_output_tokens=args.max_output_tokens,
            pass_custom_model=args.pass_custom_model,
            additional_args=args.run_args
        )
        
        if run_urls:
            logger.info(f"Successfully logged {len(run_urls)} datasets to WandB")
            for i, url in enumerate(run_urls):
                logger.info(f"  {datasets[i]}: {url}")
        
    elif args.model and datasets:
        # Log existing results for specific model/datasets
        logged_count = 0
        for dataset_name in datasets:
            result_files = find_evaluation_files(args.work_dir, args.model, dataset_name)
            if not result_files:
                logger.warning(f"No result files found for {args.model} x {dataset_name}")
                continue
                
            run_url = log_to_wandb(
                model_name=args.model,
                dataset_name=dataset_name,
                result_files=result_files,
                project=args.project,
                tags=args.tags,
                notes=args.notes
            )
            logger.info(f"Results for {dataset_name} logged to WandB: {run_url}")
            logged_count += 1
        
        if logged_count == 0:
            logger.error(f"No results found for any of the specified datasets: {datasets}")
        else:
            logger.info(f"Successfully logged {logged_count}/{len(datasets)} datasets")
        
    else:
        logger.error("Please specify either --log-all or provide --model and datasets (--data or --dataset)")
        parser.print_help()


if __name__ == "__main__":
    main()
