# Licensed under an MIT style license -- see LICENSE.md

set -e

DIR=`python -c "import pesummary; from pathlib import Path; print(Path(pesummary.__file__).parent.parent)"`
bash ${DIR}/pesummary/tests/_bilby.sh
sed -i 's/result = bilby.run_sampler(/result = bilby.run_sampler(dlogz=1000, /' ./fast_tutorial.py
python fast_tutorial.py
summarypages --webdir ./outdir/webpage --samples ./outdir/fast_tutorial_result.json ./outdir/fast_tutorial_result.hdf5 --gw --disable_expert --disable_interactive --no_ligo_skymap
