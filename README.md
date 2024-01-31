# Music Record Recommendation using Spark and Hadoop

  

## Popularity Basedline

*   The popularity of each recording is defined as follows: the total rating the recording has divided by the total number of users who have rated the recording plus beta which is a hyperparameter. The beta here serves as a damping factor to avoid unstable estimation, since the numbers of users who listened to the recording are unequally distributed. 

  

## Latent Factor Model

*   Implemented a latent factor model, by Spark's alternating least squares (ALS) method 
*   The mAP from the ALS model is larger than the mAP from the popularity baseline model, there are some improvements for the recommendation. 

  

## lightFM

*   Single-machine implementation to compare with the models ran parallelly
