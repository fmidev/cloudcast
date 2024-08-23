set -uex

labels="
unet-mae-hist=4-dt=True-topo=False-terrain=False-lc=20-oh=False-sun=False-img_size=512x512
"
#labels="
#unet-bcl1-hist=4-dt=False-topo=False-terrain=False-lc=12-oh=False-sun=True-img_size=512x512
#"


base=$HOME/cloudcast/data/official-verification/full-resolution-949/
seasons="autumn winter spring summer"
#seasons="all-seasons"
seasons="winter"
for label in $labels; do
  for season in $seasons; do
    plotdir=$base/mae/plots/$label/$season
    statdir=$base/mae/stats/$label/$season
    plotdir=/tmp
    statdir=/tmp
    mkdir -p $plotdir $statdir

    predlen=20
# meps:$base/$season/meps-$season.npz gt:$base/$season/ground-truth-$season.npz \
    python3 verify.py \
 	  --label $label \
	  --prediction_file $base/$season/mae-meps-$season.npz \
	  --include_additional meps:$base/$season/meps-$season.npz gt:$base/$season/ground-truth-$season.npz \
	  --plot_dir $plotdir \
	  --stats_dir $statdir \
	  --hourly_data \
	  --prediction_len $predlen --score fss

  done
done

