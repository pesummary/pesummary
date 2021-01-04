# Here we show how the pesummary `kdeplot` differs from the `seaborn`
# implementation and allows for added flexability regarding the choice of kde
# kernel

# First import everything
import matplotlib.pyplot as plt
from seaborn import kdeplot as seaborn_kdeplot
from pesummary.core.plots.seaborn.kde import kdeplot
import numpy as np

# Next generate some random data
x = np.random.normal(10, 2, 1000)

# make a seaborn kdeplot
ax = seaborn_kdeplot(x=x)
ll = ax.get_lines()[-1]
ll.set_label("seaborn")

# make pesummary kdeplot to show that the output is the same as the seaborn
# implementation
ax = kdeplot(x=x, ax=ax)
ll = ax.get_lines()[-1]
ll.set_label("pesummary")

# make pesummary kdeplot with custom KDE kernel. Here we use the bounded_1d_kde
# implemented within pesummary with bounds 6<=x<=14 and we will use Reflective
# boundaries
from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde
ax = kdeplot(
    x=x, ax=ax, kde_kernel=bounded_1d_kde,
    kde_kwargs={"xlow": 6, "xhigh": 14, "method": "Reflection"}
)
ll = ax.get_lines()[-1]
ll.set_label("pesummary custom kde")

plt.legend(loc="best")
plt.show()
