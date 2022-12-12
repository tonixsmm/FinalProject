# Tony Nguyen
# CPSC 222 01
# Dr. Gina Sprint
# December 13th, 2022
# This file processes the original Apple Health Dataset in XML format and convert it into a CSV.

import pandas as pd

df = pd.read_xml("export.xml")
df.to_csv("export_converted.csv")