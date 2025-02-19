# Create baseline FID score for SRGAN to evaluate how well generative network is doing
# Must be run from root directory as python -m scripts.baseline

import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data.factory import TaskFactory
from torcheval.metrics import FrechetInceptionDistance
from torch.nn.functional import interpolate

def computeFIDBaseline(dloader):
    """ Compute the average Frechet Inception Distance (FID) score over 
        a test-set using bilinear interpolation to create a baseline FID

    Args:
        dloader (torch.utils.data.dataloader): _description_
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set eval mode
    fid = FrechetInceptionDistance(device=device)
    with torch.no_grad():
        for batch in dloader:
            # Images, labels to device               
            lrs, hrs = batch
            lrs = lrs.to(device)
            hrs = hrs.to(device)
            hrs = ((hrs + 1) / 2).float() # Shift to [0, 1]

            # Upsample LR
            upsampled = interpolate(lrs,
                            scale_factor= 4,
                            recompute_scale_factor=False,
                            mode='bicubic')
            upsampled = torch.clamp(upsampled, 0, 1)
            # Convert real and fake from [-1, 1] to [0, 1]

            # Update fake, update real
            fid.update(upsampled, False) # Fake
            fid.update(hrs, True) # Real

    return fid.compute() 

# Get test set data
root_dir = os.getcwd()
dataset_name = 'imagenet'
dataset_dir = os.path.join(root_dir, "datasets/" + dataset_name + '/Data/CLS-LOC')
test_dataset = TaskFactory.createTaskDataSet(dataset_name, os.path.join(dataset_dir, \
                                             "subtest"), 4, cropsz=(256, 256))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

# View test images 
#lr, hr = test_dataset[0]
#lr = (lr * 255).byte()
#hr = (((hr + 1) / 2) * 255).byte()
#fig, axes = plt.subplots(1, 2)
#axes[0].imshow(lr.permute(1, 2, 0))
#axes[1].imshow(hr.permute(1, 2, 0))
#plt.show()

# Compute FID
ret = computeFIDBaseline(test_loader)
print(ret)


