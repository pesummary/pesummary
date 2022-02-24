# Licensed under an MIT style license -- see LICENSE.md

curl https://raw.githubusercontent.com/lscsoft/bilby/master/examples/gw_examples/injection_examples/fast_tutorial.py -o fast_tutorial.py
sed -i '/result.plot_corner()/d' ./fast_tutorial.py
echo "result.save_to_file(extension='hdf5', overwrite=True)" >> ./fast_tutorial.py
