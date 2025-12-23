set -uex

if [ -s log/train.log ]; then
  mv log/train.log log/train.log.$(date +%Y%m%d%H%M%S)
fi

export MODEL_FAMILY=pgu

# New run
# export CC2_RUN_NAME=$(sh generate-random-name.sh)
# config=config.yaml

# Resume a run
# export CC2_RUN_NAME=tan-point
# export CC2_RUN_NUMBER=$(ls ./runs/$CC2_RUN_NAME/|tail -1)
# export CC2_RUN_DIR=runs/$CC2_RUN_NAME/$CC2_RUN_NUMBER

# 1. Start from scratch
# nohup python3 cc2trainer.py fit --config $config > log/train.log 2>&1 &

# 2. Continue training from a checkpoint including optimizer state
# nohup python3 cc2trainer.py fit --config $CC2_RUN_DIR/config.yaml --ckpt_path $CC2_RUN_DIR/checkpoints/last.ckpt > log/train.log 2>&1 &

# 3. Continue training from a checkpoint excluding optimizer state
# nohup python3 cc2trainer.py fit --config $CC2_RUN_DIR/config.yaml --model.init_weights_from_ckpt=true > log/train.log 2>&1 &

# 4. Continue training from a checkpoint excluding optimizer state and adapt checkpoint to model
# nohup python3 cc2trainer.py fit --config $CC2_RUN_DIR/config.yaml --model.init_weights_from_ckpt=true --model.adapt_ckpt_resolution=true > log/train.log 2>&1 &
