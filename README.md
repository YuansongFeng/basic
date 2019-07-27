This repo contains an implementation of ResNet + Transformer from scratch. The purpose of this repo is to 
1. practice engineering and training skills, to lay the ground work for future research. 
2. try out different visualization tricks and experiments to better understand the mechanics of the two 
most state-of-art algorithms in cv and nlp.  
3. formulate a clear and robust bridge between nlp and cv. record effective ways to make use of multi-modal 
knowledge. 

Steps to take:
1. Implement ResNet and test on Places365. Use shallow version. 3 days
2. Remove different structures from ResNet to observe the performance difference. 1 day
3. Inspect feature maps of ResNet using some visualization techniques, from low to high levels. 1 day
4. Implement Transformer block and test on translation task. Use shallow version. 3 days
5. Follow existing work to generate image captions based on ResNet features, should establish a baseline. 2 days
6. Co-train the resnet with the transformer using our formulation. 3 days