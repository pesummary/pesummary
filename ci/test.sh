set -e
# Copyright (C) 2019 Charlie Hoy <charlie.hoy@ligo.org>
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

PWD=`pwd`
SCRIPT_DIR=`dirname "${BASH_SOURCE[0]}"`
COVERAGE="False"
EXPRESSION="False"
ALLOWED=(executables imports tests workflow skymap bilby lalinference GW190412 GW190425 GW190814 GWTC1 style)
IGNORE=()

for i in "$@"; do
    case $i in
        -t|--type)
        RUN="$2"
        shift 2
        ;;
        -c|--coverage)
        COVERAGE="True"
        shift
        ;;
        -i|--ignore)
        IGNORE+=("$2")
        shift 2
        ;;
        -k|--expression)
        EXPRESSION="$2"
        shift 2
        ;;
    esac
done


for type in ${ALLOWED[@]}; do
    if [[ ${type} == ${RUN} ]]; then
        VALID="True"
        break
    else
        VALID="False"
    fi
done


if [[ ${VALID} == "False" ]]; then
    echo "Invalid test type ${RUN}. Please choose one from the following: ${ALLOWED[@]}"
    exit 1
fi


cd ${SCRIPT_DIR}
cd ../


if [[ ${RUN} == "executables" ]]; then
    bash pesummary/tests/executables.sh
elif [[ ${RUN} == "imports" ]]; then
    bash pesummary/tests/imports.sh
elif [[ ${RUN} == "style" ]]; then
    flake8 .
elif [[ ${RUN} == "tests" ]]; then
    cd pesummary
    if [[ ${COVERAGE} == "True" ]]; then
        COMMAND="coverage run -m "
    else
        COMMAND=""
    fi
    COMMAND+="pytest tests "
    for ignore in ${IGNORE[@]}; do
        COMMAND+="--ignore ${ignore} "
    done
    if [[ ${EXPRESSION} != "False" ]]; then
        COMMAND+="-k '${EXPRESSION}' "
    fi
    echo ${COMMAND}
    eval ${COMMAND}
    if  [[ ${COVERAGE} == "True" ]]; then
        coverage report
        coverage html
        coverage-badge -o coverage_badge.svg -f
    fi
    cd ..
elif [[ ${RUN} == "workflow" ]]; then
    cd pesummary
    if [[ ${COVERAGE} == "True" ]]; then
        COMMAND="coverage run -m "
    else
        COMMAND=""
    fi
    COMMAND+="pytest tests/workflow_test.py "
    if [[ ${EXPRESSION} != "False" ]]; then
        COMMAND+="-k '${EXPRESSION}' "
    fi
    echo ${COMMAND}
    eval ${COMMAND}
    cd ..
elif [[ ${RUN} == "skymap" ]]; then
    cd pesummary
    pytest tests/ligo_skymap_test.py
    cd ..
elif [[ ${RUN} == "lalinference" ]]; then
    cd pesummary
    bash tests/lalinference.sh
    cd ..
elif [[ ${RUN} == "bilby" ]]; then
    cd pesummary
    bash tests/bilby.sh
    cd ..
elif [[ ${RUN} == "GW190412" ]]; then
    cd pesummary
    curl https://dcc.ligo.org/public/0163/P190412/009/posterior_samples.h5 -o GW190412_posterior_samples.h5
    python tests/existing_file.py -f GW190412_posterior_samples.h5
    cd ..
elif [[ ${RUN} == "GW190425" ]]; then
    cd pesummary
    curl https://dcc.ligo.org/public/0165/P2000026/001/GW190425_posterior_samples.h5 -o GW190425_posterior_samples.h5
    python tests/existing_file.py -f GW190425_posterior_samples.h5
    cd ..
elif [[ ${RUN} == "GWTC1" ]]; then
    cd pesummary
    curl -O https://dcc.ligo.org/public/0157/P1800370/004/GWTC-1_sample_release.tar.gz
    tar -xf GWTC-1_sample_release.tar.gz
    python tests/existing_file.py -f GWTC-1_sample_release/GW150914_GWTC-1.hdf5 -t pesummary.gw.file.formats.GWTC1.GWTC1
    summarypages --webdir ./GWTC1 --no_ligo_skymap --samples GWTC-1_sample_release/GW150914_GWTC-1.hdf5 GWTC-1_sample_release/GW170817_GWTC-1.hdf5 --path_to_samples None IMRPhenomPv2NRT_highSpin_posterior --labels GW150914 GW170817 --gw
fi


cd ${PWD}
