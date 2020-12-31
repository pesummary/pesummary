from pesummary.gw.plots.publication import violin_plots
from pesummary.gw.plots.latex_labels import GWlatex_labels
import matplotlib.pyplot as plt
from pesummary.core.plots.bounded_1d_kde import Bounded_1d_kde
import numpy as np

# Lets randomly make some samples for the tidal deformability on the secondary
# object
parameter = "lambda_2"
samples = [
    np.random.uniform(0., 3000, 1000),
    np.random.uniform(0., 3000, 1000)
]
labels = ["a", "b"]

# First lets make a violin plot which uses a Gaussian KDE
fig = violin_plots(parameter, samples, labels, GWlatex_labels)
plt.savefig("gaussian.png")
plt.close()

# Now lets make a violin plot which uses Bounded_1d_kde's, meaning that the
# KDE is bounded between 0 and 3000
fig = violin_plots(
    parameter, samples, labels, GWlatex_labels, kde=Bounded_1d_kde,
    kde_kwargs={"xlow": 0.0, "xhigh": 3000.0}, cut=0
)
plt.savefig("bounded_kde.png")
plt.close()

# Now lets show how this can be applied to a seaborn split example. See
# https://seaborn.pydata.org/generated/seaborn.violinplot.html for details. We
# choose to boound the data to be between 10 and 50
from pesummary.core.plots.seaborn.violin import violinplot
import seaborn as sns

tips = sns.load_dataset("tips")
fig = plt.figure()
ax = fig.gca()
ax = violinplot(
    x="day", y="total_bill", hue="smoker", data=tips, palette="muted",
    split=True, kde=Bounded_1d_kde, kde_kwargs={"xlow": 10.0, "xhigh": 50.0},
    ax=ax
)
plt.savefig("split.png")
plt.close()
