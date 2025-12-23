set -uex

if [ $# -eq 1 ]; then
  RUN_NAME=$1
  base=../runs/$RUN_NAME

  config=$base/config.yaml

  if [ -f "$base/checkpoints/best.ckpt" ]; then
    ckpt=$base/checkpoints/best.ckpt
  else
    ckpt=$(ls $base/checkpoints/ep* | sort -t= -k2 -n | tail -1)
  fi
elif [ $# -eq 2 ]; then
  config=$1
  ckpt=$2
else
  exit 1
fi

dir=$(dirname $(dirname $(realpath $0)))

CC2_DATA_PATH="${CC2_DATA_PATH:-$dir/data}"
CC2_OUTPUT_PATH="${CC2_OUTPUT_PATH:-$dir/output}"

export prev_atime=$(date -ud "${ANALYSIS_TIME}z 1 hours ago" +"%Y-%m-%d %H:00:00")

cd ../cc2

python3 cc2trainer.py predict \
	--config $config \
	--ckpt_path $ckpt \
	--trainer.logger=False \
	--trainer.devices=1 \
	--trainer.num_nodes=1 \
	--trainer.strategy=auto \
	--trainer.callbacks=[] \
	--data.data_path="[\"$CC2_DATA_PATH/meps-nwcsaf.zarr\"]" \
	--data.static_forcing_path="$CC2_DATA_PATH/meps-const-v4.zarr" \
	--data.batch_size=1 \
	--data.train_start=null \
	--data.train_end=null \
	--data.val_start=null \
	--data.val_end=null \
	--data.predict_start="$prev_atime" \
	--data.predict_end="$ANALYSIS_TIME" \
	--model.branch_from_run=null \
	--model.use_statistics_from_checkpoint=true \
	--model.rollout_length=$MAX_HOURS \
	--model.freeze_layers=[] \
	--model.test_output_directory=$CC2_OUTPUT_PATH
