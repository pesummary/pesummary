# Licensed under an MIT style license -- see LICENSE.md

# get the DINGO model
DINGO_PT_FILE=$(curl -k -s https://webdav.tuebingen.mpg.de/dingo-ci/ | grep -oE 'href="[^"]+\.pt"' | sed 's/href="//;s/"//' | head -n 1)
curl -k -O "https://webdav.tuebingen.mpg.de/dingo-ci/$DINGO_PT_FILE"

# download an ini file to run basic inference on 
curl https://git.ligo.org/nihar.gupte/dingo-ci/-/blob/master/GW150914.ini

# replacing the network in the ini file with the most updated network from the CI 
sed -i "s|model=tmp_model|model=${DINGO_PT_FILE}|" GW150914.ini

dingo_pipe ./GW150914.ini
FILE=`ls ./outdir_GW150914/result/*sampling.*`
summarypages --webdir ./outdir_GW150914/webpage --samples ${FILE} --gw --disable_expert --disable_interactive --no_ligo_skymap --no_conversion