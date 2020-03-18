tags=($(git tag))
stable=${tags[${#tags[@]}-1]}

mkdir -p ../stable_docs
mkdir -p ../unstable_docs
mv ./* ../unstable_docs

git checkout ${stable} ./*
mv ./* ../stable_docs
mv ../stable_docs .
mv ../unstable_docs .
cp unstable_docs/Makefile .
cp unstable_docs/conf.py .

python -c "from pesummary.gw.file.standard_names import descriptive_names; f = open('unstable_docs/data/parameter_descriptions.csv', 'w'); lines = ['{},{}\n'.format(key, item) for key, item in descriptive_names.items()]; f.writelines(lines)"

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
    <a href="stable_docs/index.html" class="btn btn-info" style="font-size: 40px; width: 400px; margin-right: 0.5em" ole="button">${stable}</a>
    <a href="unstable_docs/index.html" class="btn btn-danger" style="font-size: 40px; width: 400px; margin-left: 0.5em" role="button">Latest</a>
    </div>
</div>
</body>
EOL

cat >> index.rst <<EOL
EMPTY FILE
EOL

cd stable_docs
make html
cd ../unstable_docs
make html
cd ..
mkdir -p _build/html
mkdir -p _build/html/stable_docs
mkdir -p _build/html/unstable_docs
mv stable_docs/_build/html/* _build/html/stable_docs/
mv unstable_docs/_build/html/* _build/html/unstable_docs/
cp ../pesummary/core/webpage/copyright.txt ./_build/html
mv index.html _build/html
