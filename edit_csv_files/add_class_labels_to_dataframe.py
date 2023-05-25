# Creates new columns in csv files, for the class of a galaxy
import pandas as pd

dir = 'C:/Users/paxso/galclass_da/gal_img_full/gal_img_full/' # Directory with dataframes
df = pd.read_csv(dir+"gal_list.csv") # Dataframe with all sdss images
hsc = pd.read_csv(dir+"hsc_dataframe.csv") # Dataframe with just hsc == true images

# Add label of type of galaxy and an integer for that type to new columns, for sdss data frame
df_t = []
df_num = []
for i in range(len(df)):
    pEll = df.loc[i, ('pEll')]
    pS0 = df.loc[i, ('pS0')]
    pSab = df.loc[i, ('pSab')]
    pScd = df.loc[i, ('pScd')]
    # Use equation to get class, based on probability of it being each class
    # T fit from Meert et al. (2015)
    T = (-4.6 * pEll)-(2.4 * pS0) + (2.5*pSab) + (6.1*pScd)

    if T <= -3:
        typei = 'ell'
        typenum = 0
    elif (0.5 >= T) and (T > -3):
        typei = 's0'
        typenum = 1
    elif (4 >= T) and (T > 0.5):
        typei = 'sab'
        typenum = 2
    elif T > 4:
        typei = 'scd'
        typenum = 3

    df_t.append(typei)
    df_num.append(typenum)
# Add columns to dataframe
df['class'] = df_t
df['class_num'] = df_num

# Repeat for hsc dataframe
hsc_t = []
hsc_num = []
for i in range(len(hsc)):
    pEll = hsc.loc[i, ('pEll')]
    pS0 = hsc.loc[i, ('pS0')]
    pSab = hsc.loc[i, ('pSab')]
    pScd = hsc.loc[i, ('pScd')]
    T = (-4.6 * pEll)-(2.4 * pS0) + (2.5*pSab) + (6.1*pScd)

    if T <= -3:
        typei = 'ell'
        typenum = 0
    elif (0.5 >= T) and (T > -3):
        typei = 's0'
        typenum = 1
    elif (4 >= T) and (T > 0.5):
        typei = 'sab'
        typenum = 2
    elif T > 4:
        typei = 'scd'
        typenum = 3

    hsc_t.append(typei)
    hsc_num.append(typenum)

hsc['class'] = hsc_t
hsc['class_num'] = hsc_num
# Update files
df.to_csv(dir + 'gal_list.csv')
hsc.to_csv(dir + 'hsc_dataframe.csv')