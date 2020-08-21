# Here I will show you how to recreate the plots in GWTC1 using the
# summarypublication executable. Note that we cannot reproduce all of the
# plots in GWTC1 because the data release does not hold all of the required
# parameters

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

# Now lets assign the same colors and linestyles that were used in GWTC1
COLORS=('#00a7f0' '#9da500' '#c59700' '#55b300' '#f77b00' '#ea65ff' '#00b1a4' '#00b47d' '#ff6c91' '#00aec2' '#9f8eff')
LINESTYLES=(solid dashed solid solid dashed solid solid dashed solid dashed dashed)

# Left panel of Figure 4
summarypublication --plot 2d_contour \
                   --webdir ./GWTC-1_sample_release \
                   --samples ${FILES[@]} \
                   --labels ${LABELS[@]} \
                   --parameters mass_1 mass_2 \
                   --colors ${COLORS[@]} \
                   --linestyles ${LINESTYLES[@]} \
                   --publication_kwargs xlow:0 xhigh:100 ylow:0 yhigh:200

# Top left panel of figure 5
summarypublication --plot violin \
                   --webdir ./GWTC-1_sample_release \
                   --samples ${FILES[@]} \
                   --labels ${LABELS[@]} \
                   --parameters mass_ratio \
                   --colors ${COLORS[@]} \

# Left panel of Figure 7
summarypublication --plot 2d_contour \
                   --webdir ./GWTC-1_sample_release \
                   --samples ${FILES[@]} \
                   --labels ${LABELS[@]} \
                   --parameters theta_jn luminosity_distance \
                   --colors ${COLORS[@]} \
                   --linestyles ${LINESTYLES[@]}

# Right panel of Figure 8
summarypublication --plot 2d_contour \
                   --webdir ./GWTC-1_sample_release \
                   --samples ${FILES[@]} \
                   --labels ${LABELS[@]} \
                   --parameters luminosity_distance chirp_mass \
                   --colors ${COLORS[@]} \
                   --linestyles ${LINESTYLES[@]}
