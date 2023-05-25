# Makes new csv file that includes all hsc images that aren't blank
import pandas as pd
from PIL import Image
import numpy as np

# Directory to dataframe with hsc images
dir = 'C:/Users/paxso/galclass_da/gal_img_full/gal_img_full/'
df = pd.read_csv(dir+"gal_list.csv") 
hsc = df[df['hsc_exists'] == 1].reset_index()

for i in range(len(hsc)):
    im = Image.open(dir + 'hsc/' + str(hsc.loc[i, ('label')]) + '_hscs.jpg')
    # From testing, I found that hsc images with a standard deviation of less than 6 are always blank
    if np.std(im) < 6:
        hsc.loc[i, ('label')] = 0
    # Since the code takes a few minutes, this keeps us updated on where we are
    if i%500 == 0:
        print(i)
# Reset index to make sure each row is labeled sequentially 
hsc = hsc[hsc['label'] != 0].reset_index()
# Update file
hsc.to_csv(dir + 'hsc_dataframe.csv')