import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Defines the list of all countries
countries_df = pd.read_csv('countries.txt', delimiter = "\t",header=0)
country_ls = countries_df.iloc[:,0].tolist()
country_ls.append("Afghanistan")


#%% This section extracts images from duckduckgo
import DuckDuckGoImages as ddg

for nationality in country_ls:
    keyword = f"passport {nationality}"
    ddg.download(keyword,
                 folder=nationality,
                 max_urls=200,
                 parallel=True,
                 shuffle=True,
                 remove_folder=True)
    keyword = f"id card  {nationality}"
    ddg.download(keyword,
                 folder=nationality,
                 max_urls=200,
                 parallel=True,
                 shuffle=True,
                 remove_folder=True)


