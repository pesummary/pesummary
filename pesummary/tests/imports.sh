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

mkdir .tmp
cd .tmp
modules=($(python -c "import pkgutil; import pesummary; print(' '.join([modname for _, modname, _ in pkgutil.walk_packages(path=pesummary.__path__, prefix=pesummary.__name__+'.')]))"))
for mod in ${modules[@]}; do
    if [[ ${mod} != *"pesummary.tests"* ]]; then
        python -c "print('Importing: %s' % ('${mod}')); import ${mod}";
    fi
done
cd ..
rm -r .tmp
