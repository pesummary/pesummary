from pesummary.gw.fetch import fetch_open_data

data = fetch_open_data("GW190412")
print(data.labels)
