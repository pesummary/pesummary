from pesummary.gw.file.tgr import imrct_deviation_parameters_from_final_mass_final_spin
from pesummary.io import read

samples = read("pesummary_samples/GW150914_bilby_pesummary.dat").samples_dict

mfi = samples["inspiral"]["final_mass_non_evolved"]
mfpi = samples["postinspiral"]["final_mass_non_evolved"]
chifpi = samples["postinspiral"]["final_spin_non_evolved"]

chifi = samples["inspiral"]["final_spin_non_evolved"]

probdict = imrct_deviation_parameters_from_final_mass_final_spin(
    mfi, chifi, mfpi, chifpi, N_bins=201, vectorize=True, multi_process=4,
)

fig, _, _, _ = probdict.plot(
    "final_mass_final_spin_deviations",
    type="triangle",
    truth=[0.0, 0.0],
    cmap="YlOrBr",
    levels=[0.68, 0.95],
    smooth=2.0,
    level_kwargs={"colors": ["k", "k"]},
)
fig.savefig("imrct_GW150914.png")
