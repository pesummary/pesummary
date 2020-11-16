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

git lfs clone https://git.ligo.org/lscsoft/ROQ_data --include "**/params.dat,*/4s/**"
mkdir -p lalinference/test/
mkdir -p lalinference/lib
curl https://git.ligo.org/charlie.hoy/lalsuite/raw/master/lalinference/lib/lalinference_pipe_example.ini -o lalinference_pipe_example.ini
mv lalinference_pipe_example.ini lalinference/lib
curl https://git.ligo.org/charlie.hoy/lalsuite/raw/master/lalinference/test/lalinference_nestedSampling_integration_test.sh -o fast_tutorial.sh
mv fast_tutorial.sh lalinference/test
curl https://git.ligo.org/lscsoft/lalsuite/raw/master/lalinference/test/injection_standard.xml -o injection_standard.xml
mv injection_standard.xml lalinference/test
path=`which lalinference_pipe`
base=`python -c "print('/'.join('${path}'.split('/')[:-2]))"`
sed -i "s;s|/home/albert.einstein/opt/lalsuite/| |;s|/home/albert.einstein/opt/lalsuite/|${base}|;" lalinference/test/fast_tutorial.sh
sed -i "s;tolerance=20;tolerance=500;" lalinference/test/fast_tutorial.sh
sed -i "s;cbcBayesPostProc;summarypages;" lalinference/lib/lalinference_pipe_example.ini
bash lalinference/test/fast_tutorial.sh
