# Learning To Improve Representations by Communicating About Perspectives

This is the code repository for our paper "Learning To Improve Representations by Communicating About Perspectives", which is currently under review at NeurIPS 2021. A prelimiary version of the paper can be found [here](https://drive.google.com/file/d/12jiAi9Xqq04RYj-vWK_z2wapRaOaQdqO/view?usp=sharing), though it is subject to a lot of changes in the near future.

## At a glance
We link emergent multi-agent communication with representation learning. Using downstream task performance as a metric, we find that when agents use their representations not only to _represent_ data but also use them to communicate, the learned representations are better then if they were not encouraged to use them in communication.

**Methods:** We use multiple convolutional autoencoders and assume agents share a context, which means that they observe slightly different viewpoints of the same underlying state. We use a range of loss functions that encourage the alignment of latent representations (of the same classes) during training. 

**Results:** We find that when aligning representations, performance on downstream tasks is significantly better. This is the case **only** when agents receive different viewpoints of the underlying state. We also find that population size has a significant effect on downstream task performance, but only when perspectives are used.

**Significance:** We believe that the link between multi-agent learning and representation learning is a promising one and researchers have so far only scratched the surface of what we believe is a rich and fruitful inteaction that comes almost for free when considering settings of language-based (reinforcement) learning.


For more information, please have a look at our paper!
