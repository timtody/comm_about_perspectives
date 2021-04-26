import matplotlib.pyplot as plt
from mnist import MNISTDataset


ds = MNISTDataset()
ims = ds.sample_digit(9, 2)

fig, axes = plt.subplots(ncols=2)

for im, ax in zip(ims, axes):
    ax.imshow(im.reshape(28, 28))
    ax.set_axis_off()


plt.tight_layout()
plt.show()