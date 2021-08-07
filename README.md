# Learning To Improve Representations by Communicating About Perspectives

This is the code repository for our paper "Learning To Improve Representations by Communicating About Perspectives", which is currently under review at NeurIPS 2021. A prelimiary version of the paper can be found [here](https://drive.google.com/file/d/12jiAi9Xqq04RYj-vWK_z2wapRaOaQdqO/view?usp=sharing), though it is subject to a lot of changes in the near future.

## Summary
At the core of artificial agents lie their representations of the world around them. Inpsired by research that views language as a cognitive tool as well as works that link population size to simplicity in language, we hypothesise that such a link can be shown in an emergent communication setup and then later exploited for downstream tasks. By highlighting this link between emergent communication, language-based (reinforcement) learning, and representation learning, we hope to inspire more researchers to explore the complex interactions between these fields. Furthermore, we want to highlight a specific aspect intrinsic to multi-agent systems that may be exploited to improve representations: perspectives. We believe that in settings of shared attention (for example when solving cooperative tasks), agents benefit from aligning their internal representations with those of other agents experiencing the same shared context. Because agents within the same context most likey observe a similar, but distinctive view of the focal point of shared attention, they can benefit by "agreeing" on representations. In this way, we hope to use the inherent properties of multi-agent systems to bias the systems towards generating asbtract representations.

## Architecture
<img src="prod/arch.png" alt="drawing" width="500"/>

## Results
<img src="prod/swap.png" alt="drawing" width="500"/>

<img src="prod/persp.png" alt="drawing" width="500"/>
