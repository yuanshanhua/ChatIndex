# sft projector
ASCEND_RT_VISIBLE_DEVICES=0 uv run scripts/sft_projector.py \
workloads/index_eab/10q/job+tpch.2000w.5-15q.workload.labeled.json \
--eval-set-percent 0.1 \
--output-dir saves/sft_projector \
--mm-flm \
--mm \
--mm-embedding-files embeddings/imdb.tapas.1000.safetensors embeddings/tpch.tapas.1000.safetensors \
--mm-projector-trainable \
--mm2 \
--mm2-embedding-files workloads/index_eab/job_1500.safetensors workloads/index_eab/tpch_2000.safetensors \
--mm2-projector-trainable \
--epochs 20 \
--batch-size 8 \
--learning-rate 1e-4 \
--debug config/sft-projector.yaml

# sft projector - only column projector
CUDA_VISIBLE_DEVICES=0 uv run scripts/multi_sft.py \
workloads/index_eab/10q/job+tpch.2000w.5-15q.workload.labeled.json \
--eval-set-percent 0.1 \
--output-dir saves/sft_projector_columns_only \
--mm --mm-flm \
--mm-embedding-files embeddings/imdb.tapas.1000.safetensors embeddings/tpch1g.tapas.10000.safetensors \
--mm-projector-trainable \
--epochs 20 \
--batch-size 8 \
--learning-rate 1e-4 \
--debug config/sft-projector.yaml

# sft [lm + column projector] with sft_projector.py
CUDA_VISIBLE_DEVICES=1 uv run scripts/sft_projector.py \
workloads/index_eab/10q/job+tpch.2000w.5-15q.workload.labeled.json \
--adapter-dir saves/sft_6000/checkpoint-300 \
--eval-set-percent 0.1 \
--output-dir saves/sft \
--mm \
--mm-embedding-files embeddings/imdb.tapas.1000.safetensors embeddings/tpch1g.tapas.10000.safetensors \
--mm-projector-trainable \
--mm-projector-checkpoint saves/sft_projector_columns_only/projector_step_5100.safetensors \
--epochs 20 --batch-size 2 --learning-rate 1e-4 \
--save-steps 100 --log-steps 10 \
--debug  config/lib-sft.yaml

# sft [lm + 2 column projectors] from checkpoint
CUDA_VISIBLE_DEVICES=2 uv run scripts/sft_projector.py \
workloads/index_eab/10q/dsb.7000.2000w.5-15q.workload.labeled.json \
--eval-set-percent 0.1 \
--output-dir saves/sft_6000_15000_dsb_stable \
--mm --mm-embedding-files embeddings/dsb.tapas.1000.safetensors \
--mm-projector-trainable \
--mm2 --mm2-embedding-files workloads/index_eab/dsb_7000.safetensors \
--mm2-projector-trainable \
--epochs 20 --batch-size 8 --learning-rate 1e-4 \
--debug config/sft-projector.yaml \
--adapter-dir saves/sft_6000/checkpoint-15000 \
--mm-pckpt saves/sft_6000/checkpoint-15000/projector.safetensors \
--mm2-pckpt saves/sft_6000/checkpoint-15000/projector.safetensors

# mm rl
uv run scripts/train.py \
workloads/index_eab/10q/job+tpch.2000w.5-15q.workload.json \
config/train-qwen-mm.yaml \
--adapter-dir saves/qwen2.5-7b-Instruct-sft/checkpoint-300 \
--eval-set-percent 0.1 \
--eval-batch-size 32 \
--mm --mm-embedding-files embeddings/imdb.tapas.1000.safetensors embeddings/tpch1g.tapas.10000.safetensors \
--mm-projector-trainable \
--mm-pckpt saves/qwen2.5-7b-Instruct-sft/checkpoint-300/projector.safetensors \
--mm2 \
--mm2-embedding-files workloads/index_eab/job_1500.safetensors workloads/index_eab/tpch_2000.safetensors \
--mm2-projector-trainable \
--db imdb tpch1g --dbpswd your_password \
--debug


# compare projector
uv run scripts/utils/compare_safetensors.py \
saves/qwen2.5-7b-Instruct-sft-projector/checkpoint-100/model.safetensors \
saves/qwen2.5-7b-Instruct-sft-projector/checkpoint-300/model.safetensors


# evaluate 
# 通过 yaml 指定 lora adapter. 如果启用 mm 且未指定 --mm-pckpt, 则将从 adapter 同目录加载 projector. 否则使用 --mm-pckpt 指定的固定 projector
uv run scripts/eval.py workloads/index_eab/10q/job.1500.1000w.5-15q.workload.json \
eval/res1.json \
--mm \
--mm-embedding-files embeddings/imdb.tapas.1000.safetensors embeddings/tpch1g.tapas.10000.safetensors \
--mm-pckpt saves/qwen2.5-7b-Instruct-sft/checkpoint-300/projector.safetensors \
-b 4 --gpu 0,1 \
--db imdb --dbpswd your_password \
--debug  config/eval-qwen.yaml

# eval both modals
uv run scripts/eval.py workloads/index_eab/10q/tpch.2000.1000w.5-15q.workload.json \
eval/res1.json \
--mm --mm2 \
--mm-embedding-files embeddings/imdb.tapas.1000.safetensors embeddings/tpch1g.tapas.10000.safetensors \
--mm2-embedding-files workloads/index_eab/job_1500.safetensors workloads/index_eab/tpch_2000.safetensors \
-b 4 --gpu 1 \
--db imdb tpch1g --dbpswd your_password \
--debug  config/eval-qwen.yaml

# generate table embeddings
uv run scripts/gen/column_embeddings.py \
--db tpch1g \
--dbpswd your_password \
--sample_count 1000 \
--output embeddings/tpch.tapas.1000.safetensors


# generate workloads
uv run scripts/gen/workloads.py \
workloads/index_eab/tpch_2000.sql \
workloads/index_eab/ \
tpch.2000q.1000w.1q.300c \
--min-size 1 --max-size 1 \
-n 1000 \
--db tpch1g --dbuser surunze --dbport 5433 \
--debug


# merge workloads
uv run scripts/gen/merge.py  workloads/index_eab/1q/*.json  workloads/index_eab/1q/tpch_imdb.2000w.1q.workload.json

# 强制使用行缓冲输出日志
stdbuf -oL command 2>&1 | tee output.log
# 打印 npu 显存使用情况
stdbuf -oL npu-smi info watch -s m 2>&1 | tee npu-smi.log
# 同时打印时间到文件
stdbuf -oL npu-smi info watch -s m 2>&1 | tee >(awk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}' > npu-smi.log)
# gpu
nvidia-smi dmon -i 0,1 -s m | tee nvidia-smi.log