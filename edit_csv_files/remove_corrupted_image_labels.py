# Remove the labels for images that are corrupted
import pandas as pd
from PIL import Image

# Directory to dataframe with all sdss images (df) and hsc images (hsc)
dir = 'C:/Users/paxso/galclass_da/gal_img_full/gal_img_full/'
df = pd.read_csv(dir+"gal_list.csv") 
hsc = pd.read_csv(dir+"hsc_dataframe.csv")
# Loop through all images - if you can open them do nothing. If you can't, set the label to 0
for i in range(len(df)):
    try:
        im = Image.open(dir + str(df.loc[i, ('label')] + '_sdss.jpg'))
    except:
        df.loc[i, ('label')] = 0
        print(i)
# Get rid of all the rows where the label was set to 0
df = df[df['label'] != 0]
# Reset the index if possible
try:
    df = df.reset_index()
except:
    pass
# Repeat the last steps for the hsc dataframe
for i in range(len(hsc)):
    try:
        im = Image.open(dir + 'hsc/' + str(hsc.loc[i, ('label')] + '_hscs.jpg'))
    except:
        hsc.loc[i, ('label')] = 0
        print(i)

hsc = hsc[hsc['label'] != 0]
try:
    hsc = hsc.reset_index()
except:
    pass
# Update files
df.to_csv(dir + 'gal_list.csv')
hsc.to_csv(dir + 'hsc_dataframe.csv')