import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import random
random.seed(222222)
torch.manual_seed(222222)
np.random.seed(222222)

class GalaxyDataset(Dataset):
        ''' 
        When indexed, returns a tuple with a pillow image and the label (as a long tensor) by index
        '''
        def __init__(self, csv_file, root_dir, domain, transform=None):
            """
            Arguments:
                df (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                domain (string saying source or target): Defines the domain
                    on a sample.
            """
            self.df = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform
            self.domain = domain
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, i):
            if torch.is_tensor(i):
                i = i.tolist()
            if self.domain == 'source':
                img_name = self.root_dir + str(self.df['label'][i]) + '_sdss.jpg'
            if self.domain == 'target':
                img_name = self.root_dir + 'hsc/' + str(self.df['label'][i]) + '_hscs.jpg'
                
            image = Image.open(img_name)
            label = self.df.loc[i, ('class_num')]
            # Put the label in the correct data type
            label = torch.from_numpy(np.array([label]).astype('float')).type(torch.LongTensor)

            if self.transform:
                image = self.transform(image)

            sample = (image, label)
            
            return sample
    
class Rotate(object):
    '''
    Randomly rotates an image by 90 degrees
    '''
    def __call__(self, image):
        angle = random.choice([-90., 0., 90., 180.])
        image = transforms.functional.rotate(image, angle)
        return image
# Generator makes sure whenever we split train, val, and test, we get the same images in each
generator1 = torch.Generator().manual_seed(22) 
# Downsizes the image, randomly flips it horizontally, randomly rotates by 90 degrees, puts it into tensor form
transforms_set = transforms.Compose([transforms.Resize((64,64)), transforms.RandomHorizontalFlip(p=0.5), 
                                     Rotate(), transforms.ToTensor()])

def get_dataloader(csv_file, root_dir, domain, batch_size, train_size=.80, val_size=.10, test_size=.10):
    '''
    Input:
        csv_file - string, the path to the csv file
        root_dir - string, the directory where the images are held 
        domain - string, either "source" or "target" to define domain
        batch_size - int batch size
        train_size, val_size, test_size - floats that add to one
    Returns:
        Two tuples, the first with train/val/test dataloaders, and the second with train/val/test datasets
    '''
    dataset = GalaxyDataset(csv_file, root_dir, domain, transforms_set) # Initiate the dataset
    # Get the number of samples for train, val, and test                        
    train_size = int(len(dataset) * train_size)
    val_size = int(len(dataset) * val_size)
    test_size = int(len(dataset) * test_size)
    extra = len(dataset) - (train_size + val_size + test_size) # make sure the sum of the train, val, and test size is the size of the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size+extra, val_size, test_size],
                                                            generator=generator1)
    # Initiate the pytorch dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, 
                                  drop_last=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, 
                                drop_last=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, 
                                 drop_last=True, num_workers=8)
    
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    datasets = (train_dataset, val_dataset, test_dataset)

    return dataloaders, datasets