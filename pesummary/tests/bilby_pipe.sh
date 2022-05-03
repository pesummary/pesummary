# Licensed under an MIT style license -- see LICENSE.md

set -e

curl https://git.ligo.org/lscsoft/bilby_pipe/-/raw/master/examples/event/GW150914.ini -o GW150914.ini
sed -i 's/coherence-test = True/coherence-test = False/' ./GW150914.ini
sed -i 's/sampler-kwargs = /sampler-kwargs = {"nlive": 1000, "dlogz": 1000} #/' ./GW150914.ini
sed -i 's/channel-dict = {H1:DCS-CALIB_STRAIN_C02, L1:DCS-CALIB_STRAIN_C02}/channel-dict = {H1:GWOSC, L1:GWOSC}/' ./GW150914.ini
sed -i 's/create-plots = True/create-plots = False\nresult-format = json/' ./GW150914.ini
sed -i 's/create-summary = True/create-summary = False/' ./GW150914.ini
sed -i 's/n-parallel = 4/n-parallel = 2/' ./GW150914.ini

bilby_pipe ./GW150914.ini
bash outdir_GW150914/submit/bash_GW150914.sh
summarypages --webdir ./outdir_GW150914/webpage --samples ./outdir_GW150914/result/GW150914_data0_1126259462-4_analysis_H1L1_merge_result.json --gw --disable_expert --disable_interactive --no_ligo_skymap
