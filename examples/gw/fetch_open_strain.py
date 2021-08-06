from pesummary.gw.fetch import fetch_open_strain
import os

# Gravitational wave strain data is typically either 32s or 4096s in duration
# with a sampling rate of either 4KHz or 16KHz. We can download either by simply
# specifying the event name and specifying the duration and sampling rate with
# the `duration` and `sampling_rate` kwargs respectively
path = fetch_open_strain(
    "GW190412", IFO="L1", duration=32, sampling_rate=4096, read_file=False
)
print(path)

# If we wish to read the file, we need to specify the channel name
data = fetch_open_strain(
    "GW190412", IFO="L1", duration=32, sampling_rate=4096, read_file=True,
    channel="L1:GWOSC-4KHZ_R1_STRAIN"
)
print(data)
