TMPDIR=${TMPDIR:-/tmp}
export IN_CONTEXT_DATA_PATH="/tmp"

# Regression Context Size
for context_size in 1 5 10 20 30 40 50 60; do
  exp_name="memory_regress_scale_ctx_size"
  for dataset_size in 10000 30000 50000 70000 90000; do
    num_heads=20
    python3 src/main.py --cfg.task_args.is_classification=False --cfg.task_args.n_classes=1 --cfg.log_prefix=${exp_name}/ctx/${context_size}/dataset/${dataset_size} --cfg.train_size=${dataset_size} --cfg.task_args.context_size=${context_size} --cfg.model_args.num_heads=${num_heads}
  done
  exp_name="memory_regress_scale_ctx_size_shared"
  for dataset_size in 10000 15000 20000 25000; do
    num_heads=20
    context_type="fixed"
    python3 src/main.py --cfg.task_args.context_type=${context_type} --cfg.task_args.is_classification=False --cfg.task_args.n_classes=1 --cfg.log_prefix=${exp_name}/ctx/${context_size}/dataset/${dataset_size} --cfg.train_size=${dataset_size} --cfg.task_args.context_size=${context_size} --cfg.model_args.num_heads=${num_heads}
 done
done

# Regression Number of Heads
for num_heads in 1 5 10 20 30 40 50 60; do
  exp_name="memory_regress_scale_head"
  for dataset_size in 10000 30000 50000 70000 90000; do
    context_size=32
    python3 src/main.py --cfg.task_args.is_classification=False --cfg.task_args.n_classes=1 --cfg.log_prefix=${exp_name}/heads/${num_heads}/dataset/${dataset_size} --cfg.train_size=${dataset_size} --cfg.task_args.context_size=${context_size} --cfg.model_args.num_heads=${num_heads}
  done
  exp_name="memory_regress_scale_head_shared"
  for dataset_size in 10000 15000 20000 25000 30000; do
    context_size=32
    context_type="fixed"
    python3 src/main.py --cfg.task_args.context_type=${context_type} --cfg.task_args.is_classification=False --cfg.task_args.n_classes=1 --cfg.log_prefix=${exp_name}/heads/${num_heads}/dataset/${dataset_size} --cfg.train_size=${dataset_size} --cfg.task_args.context_size=${context_size} --cfg.model_args.num_heads=${num_heads}
  done
done

# Classification Number of Heads
for num_heads in 1 5 10 20 30 40 50 60; do
  exp_name="memory_scale_head"
  for dataset_size in 30000 50000 70000 90000 110000 130000; do
    context_size=32
    python3 src/main.py --cfg.log_prefix=${exp_name}/heads/${num_heads}/dataset/${dataset_size} --cfg.train_size=${dataset_size} --cfg.task_args.context_size=${context_size} --cfg.model_args.num_heads=${num_heads}
  done
  exp_name="memory_scale_head_shared"
  for dataset_size in 10000 20000 30000 40000; do
    context_size=32
    context_type="fixed"
    python3 src/main.py --cfg.task_args.context_type=${context_type} --cfg.log_prefix=${exp_name}/heads/${num_heads}/dataset/${dataset_size} --cfg.train_size=${dataset_size} --cfg.task_args.context_size=${context_size} --cfg.model_args.num_heads=${num_heads}
  done
done

# Classification Context Size
for context_size in 1 5 10 20 30 40 50 60; do
  exp_name="memory_scale_ctx_size"
  for dataset_size in 20000 30000 40000 50000 60000; do
    num_heads=20
    python3 src/main.py --cfg.log_prefix=${exp_name}/ctx/${context_size}/dataset/${dataset_size} --cfg.train_size=${dataset_size} --cfg.task_args.context_size=${context_size} --cfg.model_args.num_heads=${num_heads}
  done
  exp_name="memory_scale_ctx_size_shared"
  for dataset_size in 6000 10000 14000 18000; do
    num_heads=20
    context_type="fixed"
    python3 src/main.py --cfg.task_args.context_type=${context_type} --cfg.log_prefix=${exp_name}/ctx/${context_size}/dataset/${dataset_size} --cfg.train_size=${dataset_size} --cfg.task_args.context_size=${context_size} --cfg.model_args.num_heads=${num_heads}
 done
done

# Classification dH
for dh in 10 15 20 25 30 35 40 45 50 55 60 64; do
  num_heads=30
  exp_name="memory_scale_dh_shared"
  for dataset_size in 10000 15000 20000; do
    context_size=32
    context_type="fixed"
    python3 src/main.py --cfg.task_args.context_type=${context_type} --cfg.log_prefix=${exp_name}/dh/${dh}/dataset/${dataset_size} --cfg.min_train_steps=300000 --cfg.dh=${dh} --cfg.train_size=${dataset_size} --cfg.task_args.context_size=${context_size} --cfg.model_args.num_heads=${num_heads}
  done
done

# Softmax Saturation
python3 src/main.py --cfg.log_prefix=head_saturation_v3/heads/4 --cfg.include_analysis=True --cfg.lr=0.001 --cfg.min_train_steps=300000 --cfg.task_args.context_type=fixed --cfg.train_size=550 --cfg.task_args.context_size=32 --cfg.model_args.num_heads=4 --cfg.early_stop=False
python3 src/main.py --cfg.log_prefix=head_saturation_v3/heads/8 --cfg.include_analysis=True --cfg.lr=0.001 --cfg.min_train_steps=300000 --cfg.task_args.context_type=fixed --cfg.train_size=1300 --cfg.task_args.context_size=32 --cfg.model_args.num_heads=8 --cfg.early_stop=False
