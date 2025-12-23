set -xu

runs="hushed-volume/2 wary-voice/1 weathered-heron/1 seasoned-pass/1 excited-bass/1 "

python3 verify.py \
	--run_name $runs \
	--score mae mae2d psd fss variance_ratio highk_power_ratio spectral_coherence change_metrics
