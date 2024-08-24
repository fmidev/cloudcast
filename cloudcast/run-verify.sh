set -uex

labels="
unet-mae-hist=4-dt=True-topo=False-terrain=False-lc=20-oh=False-sun=False-img_size=512x512
"

export WRITE_RESULTS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

score=psd
base=$HOME/cloudcast/data/official-verification/full-resolution-949/
baselo=$HOME/cloudcast/data/official-verification/
seasons="autumn winter spring summer all-seasons"

for label in $labels; do
  for season in $seasons; do
    plotdir=$base/mae/plots/$label/$season
    statdir=$base/mae/stats/$label/$season
    plotdir=/tmp
    statdir=/tmp
    mkdir -p $plotdir $statdir

    if [ "$score" = "maess" ]; then
      python3  verify.py \
 	  --label $label \
	  --prediction_file $base/$season/mae-meps-$season.npz \
	  --include_additional meps:$base/$season/meps-$season.npz gt:$base/$season/ground-truth-$season.npz \
	  --plot_dir $plotdir \
	  --hourly_data \
	  --prediction_len 20 --score $score
    elif [ "$score" = "psd" ]; then
      # Hi-res -- only for meps
#      python3 verify.py \
# 	  --label $label \
#	  --prediction_file $base/$season/mae-meps-$season.npz \
#	  --include_additional meps:$base/$season/meps-$season.npz gt:$base/$season/ground-truth-$season.npz \
#	  --plot_dir $plotdir \
#	  --hourly_data \
#	  --prediction_len 20 --score $score
      # Lo-res -- cloudcast
      python3 verify.py \
	  --label $label \
	  --prediction_file $baselo/$season/mae-meps-$season.npz \
	  --hourly_data \
	  --prediction_len 20 --score $score
    elif [ "$score" = "fss" ]; then
      if [ "$season" != "all-seasons" ]; then
          # low res for seasonal data: 512/5 hours
          python3 verify.py \
 	  --label $label \
	  --prediction_file $baselo/$season/mae-meps-$season.npz \
	  --hourly_data \
	  --prediction_len 20 --score fss
      else
          # low res for all-seasons: 512/300 minutes
          python3 verify.py \
 	  --label $label \
	  --prediction_file $baselo/$season/mae-$season.npz \
	  --prediction_len 20 --score fss
      fi
    fi
  done
done

