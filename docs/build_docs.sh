# Copyright (C) 2020  Charlie Hoy <charlie.hoy@ligo.org>
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

set -e

# get latest tag, use current branch name if no tags found
stable=$(git describe --abbrev=0 || git rev-parse --abbrev-ref HEAD)

# get pesummary version from IGWN Conda Distribution
igwn_yaml="https://computing.docs.ligo.org/conda/environments/linux/igwn-py38.yaml"
igwn_version=$(python -c "
import requests, yaml;
py38 = yaml.safe_load(requests.get('${igwn_yaml}').content)['dependencies'];
print(list(filter(lambda x: x.startswith('pesummary='), py38))[0].split('=')[1]);
")

mkdir -p ../broken_links
mkdir -p ../stable_docs
mkdir -p ../unstable_docs
mkdir -p ../igwn_pinned
mv ./* ../unstable_docs

git checkout v0.3.4 ./*
mv ./* ../broken_links
git checkout ${stable} ./*
mv ./* ../stable_docs
cp ../unstable_docs/Makefile ../stable_docs
git checkout v${igwn_version} ./*
mv ./* ../igwn_pinned
cp ../unstable_docs/Makefile ../igwn_pinned
mv ../broken_links/ .
mv ../stable_docs .
mv ../unstable_docs .
mv ../igwn_pinned .
cp unstable_docs/Makefile .
cp unstable_docs/conf.py .

cat >> index.html <<EOL
<html lang='en'>
    <title>PESummary</title>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css'>
</head>
<body style='background-color:#F8F8F8; margin-top:5em; min-height: 100%'>
<script src='https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js'></script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js'></script>
<script src='https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js'></script>
<body>
<a href="https://git.ligo.org/lscsoft/pesummary">
    <img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/a6677b08c955af8400f44c6298f40e7d19cc5b2d/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f677261795f3664366436642e706e67" alt="Fork me on GitLab" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_gray_6d6d6d.png">
</a>
<div class="row justify-content-center">
<object data="copyright.txt" type="text/plain" width="650" style="height: 300px">
<a href="copyright.txt">No Support?</a>
</object>
</div>
<div class="button-box col-lg-12">
    <div class="row justify-content-center">
    <a class="btn btn-info" href="igwn_pinned/index.html" style="font-size: 32px; width: 300px; margin-right: 0.5em" role="button" data-toggle="tooltip" title="Version available in the igwn-py37 conda environment">igwn-py37: v${igwn_version}</a>
    <a class="btn btn-info" href="stable_docs/index.html" style="font-size: 32px; width: 300px; margin-right: 0.5em" role="button" data-toggle="tooltip" title="Latest released version on pypi">${stable}</a>
    <a class="btn btn-danger" href="unstable_docs/index.html" style="font-size: 32px; width: 300px; margin-right: 0.5em" role="button" data-toggle="tooltip" title="Latest development build">Latest</a>
    </div>
</div>
</body>
EOL

cat >> index.rst <<EOL
EMPTY FILE
EOL

cd stable_docs
python -c "print(open('index.rst', 'r').read().replace('.. warning::\n', '.. warning::'))" | perl -00ne 'print unless /.. warning::/'s > index.rst_tmp
rm index.rst
mv index.rst_tmp index.rst
make html
cd ../unstable_docs
make html
cd ../igwn_pinned
python -c "print(open('index.rst', 'r').read().replace('.. warning::\n', '.. warning::'))" | perl -00ne 'print unless /.. warning::/'s > index.rst_tmp
rm index.rst
mv index.rst_tmp index.rst
make html
cd ..
mkdir -p _build/html
mkdir -p _build/html/stable_docs
mkdir -p _build/html/unstable_docs
mkdir -p _build/html/igwn_pinned
mv stable_docs/_build/html/* _build/html/stable_docs/
mv unstable_docs/_build/html/* _build/html/unstable_docs/
mv igwn_pinned/_build/html/* _build/html/igwn_pinned/

cd broken_links
cp -r ./* ../_build/html/
BROKEN_FILES=($(find ../_build/html/ -name "*.rst" -type f -print))
for file in ${BROKEN_FILES[@]}; do
    rm ${file}
    html_name=`python -c "print('${file}'.replace('.rst', '.html'))"`
    map=`python -c "raw_path = '${html_name}'.replace('../_build/html/', ''); raw_path = raw_path.replace('.//', ''); length = len(raw_path.split('/')[:-1]); print('/'.join(['../']*length + ['stable_docs'] + raw_path.split('/')))"`
    cat >> ${html_name} <<EOL

<!DOCTYPE html>
<html>
<body onload=myFunction()>
<script>
function myFunction() {
    window.location = "${map}";
}
</script>
</body>
</html>
EOL
done

cd ..
cp ../pesummary/core/webpage/copyright.txt ./_build/html
mv index.html _build/html
