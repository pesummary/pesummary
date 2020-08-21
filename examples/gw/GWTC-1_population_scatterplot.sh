# Here we will show you how to create a population scatter plot using the
# summarypublication executable for the GWTC1 data release.

# First, we need to download the data from the DCC
curl -O https://dcc.ligo.org/public/0157/P1800370/004/GWTC-1_sample_release.tar.gz
tar -xf GWTC-1_sample_release.tar.gz

# Lets remove the priorChoice files as they do not contain posterior samples
FILES_TO_REMOVE=($(ls ./GWTC-1_sample_release/*priorChoices*))
for i in ${FILES_TO_REMOVE[@]}; do
    rm ${i}
done

# Now lets get a list of all of the files
FILES=($(ls ./GWTC-1_sample_release/*_GWTC-1.hdf5))

# Now lets get a list of labels
LABELS=()
for i in ${FILES[@]}; do
    label=`python -c "print('${i}'.split('_GWTC-1.hdf5')[0])"`
    LABELS+=(`python -c "print('${label}'.split('./GWTC-1_sample_release/')[1])"`)
done

# Now lets make the population plot
summarypublication --plot population_scatter_error \
                   --webdir ./GWTC-1_sample_release \
                   --samples ${FILES[@]} \
                   --labels ${LABELS[@]} \
                   --parameters total_mass redshift \
