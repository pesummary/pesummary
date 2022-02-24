# Licensed under an MIT style license -- see LICENSE.md

set -e

DIR=`python -c "import pesummary; from pathlib import Path; print(Path(pesummary.__file__).parent.parent)"`
bash ${DIR}/pesummary/tests/_bilby.sh
sed -i 's/result = bilby.run_sampler(/result = bilby.run_sampler(nsamples=5, printdt=5, /' ./fast_tutorial.py
sed -i 's/outdir = /outdir = "outdir_mcmc" #/' ./fast_tutorial.py
sed -i 's/dynesty/bilby_mcmc/' ./fast_tutorial.py
python fast_tutorial.py
summarypages --webdir ./outdir_mcmc/webpage --samples ./outdir_mcmc/fast_tutorial_result.json ./outdir_mcmc/fast_tutorial_result.hdf5 --gw --disable_expert --disable_interactive --no_ligo_skymap
