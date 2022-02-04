# Copyright (C) 2019 Charlie Hoy <charlie.hoy@ligo.org> This program is free
# software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

set -e

curl https://raw.githubusercontent.com/lscsoft/bilby/master/examples/gw_examples/injection_examples/fast_tutorial.py -o fast_tutorial.py
sed -i '/result.plot_corner()/d' ./fast_tutorial.py
sed -i 's/result = bilby.run_sampler(/result = bilby.run_sampler(dlogz=1000, /' ./fast_tutorial.py
echo "result.save_to_file(extension='hdf5', overwrite=True)" >> ./fast_tutorial.py
python fast_tutorial.py
summarypages --webdir ./outdir/webpage --samples ./outdir/fast_tutorial_result.json ./outdir/fast_tutorial_result.hdf5 --gw --disable_expert --disable_interactive --no_ligo_skymap
