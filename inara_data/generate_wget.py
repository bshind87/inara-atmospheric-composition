# Generate wget script for 10,000 INARA CSV files
base_url = "https://exoplanetarchive.ipac.caltech.edu/work/TMP_JFPw4G_32644/FDL/PSG/data"

with open("inara_data/wget_10000_models.sh", "w") as f:
    f.write("#!/bin/sh\n")
    for i in range(10000):
        sid = f"{i:07d}"
        dir1 = f"dir_{i//100000:02d}"
        dir2 = f"dir_{(i//1000)%100:02d}"
        url = f"{base_url}/{dir1}/{dir2}/{sid}/{sid}_spectrum_components.csv"
        f.write(f"wget -O inara_data/{sid}.csv {url}\n")

print("Script generated: inara_data/wget_10000_models.sh")
print("Total files: 10,000")
