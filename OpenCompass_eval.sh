python scripts/wandb_logger.py --run-and-log --model Qwen2.5-VL-7B-Instruct --data MathVista MathVision MathVerse DynaMath WeMath
LogicaVista --work-dir "./output" --max-output-tokens 8192 --use-vllm --batch-size 8

python run.py --data MathVista MathVision MathVerse DynaMath WeMath LogicaVista --input-dir "./outputs/Qwen2.5-VL-7B-Instruct/T20251107_G1d983a41" --llm-backend openai --model gpt-4o-mini  --verbose