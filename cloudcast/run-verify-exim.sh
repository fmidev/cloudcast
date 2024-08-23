set -uex

labels="
unet-mae-hist=4-dt=True-topo=False-terrain=False-lc=20-oh=False-sun=False-img_size=512x512
"


base=$HOME/cloudcast/data/official-verification/full-resolution-949
seasons="spring summer all-seasons"
seasons="all-seasons"
for label in $labels; do
  for season in $seasons; do
    plotdir=$base/mae/plots/$label/$season
    statdir=$base/mae/stats/$label/$season
    plotdir=/tmp
    statdir=/tmp
    mkdir -p $plotdir $statdir

    python3 verify.py \
 	  --label $label \
	  --prediction_file $base/$season/mae-exim-$season.npz \
	  --include_additional exim:$base/$season/exim-$season.npz gt:$base/$season/ground-truth-15min-$season.npz \
	  --plot_dir $plotdir \
	  --stats_dir $statdir \
	  --hourly_data \
	  --prediction_len 4 --score psd

  done
done

