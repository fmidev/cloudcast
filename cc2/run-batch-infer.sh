set -xe

if [ -z "$RUN_NAME" ]; then
  export RUN_NAME=wary-voice
  export RUN_NUMBER=1
fi

export ROLLOUT_LENGTH=5

if [ -f "/data/runs/$RUN_NAME/$RUN_NUMBER/checkpoints/best.ckpt" ]; then
  ckpt=/data/runs/$RUN_NAME/$RUN_NUMBER/checkpoints/best.ckpt
else
  ckpt=$(ls /data/runs/$RUN_NAME/$RUN_NUMBER/checkpoints/ep* | sort -t= -k2 -n | tail -1)
fi

python3 cc2trainer.py test \
	--config /data/runs/$RUN_NAME/$RUN_NUMBER/config.yaml \
	--ckpt_path $ckpt \
	--model.init_weights_from_ckpt=False \
	--model.adapt_ckpt_resolution=False \
	--data.data_path='["/anemoi-data/nwcsaf-475x535-202507-v3-anemoi.zarr"]' \
	--trainer.logger=False \
	--trainer.devices=1 \
	--trainer.num_nodes=1 \
	--trainer.strategy=auto \
	--data.train_start=null \
	--data.train_end=null \
	--data.val_start=null \
	--data.val_end=null \
	--data.test_start="2025-07-01" \
	--data.test_end="2025-08-01" \
	--data.batch_size=2 \
	--model.rollout_length=$ROLLOUT_LENGTH
