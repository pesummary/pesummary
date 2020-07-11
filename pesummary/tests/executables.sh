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

SCRIPT_DIR=`dirname "${BASH_SOURCE[0]}"`

_executables=($(ls ${SCRIPT_DIR}/../cli/summary*))
executables=()
for i in ${_executables[@]}; do
    executables+=(`python -c "print('${i}'.split('/')[-1].split('.py')[0])"`);
done
echo ${executables[@]}
for i in ${executables[@]}; do
    eval '${i} --help';
done
