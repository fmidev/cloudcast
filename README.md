# CloudCast v2

CloudCast v2 is the next version of CloudCast, a neural network for total cloud cover prediction. CloudCast is used operationally at FMI to provide short term cloud cover nowcasts.

![cloudcast example forecast](https://raw.githubusercontent.com/fmidev/cloudcast/main/cloudcast.gif)

# Description

CloudCast v2 is a spatio-temporal sliding-window vision transformer with a U-shaped encoder-decode architecture. The model is trained to predict future cloud cover based on past satellite images.

The model has been pre-trained with CERRA dataset, and then fine-tuned with MEPS+NWCSAF effective cloudiness data. The model resolution is 5km and domain matches MEPS domain (Scandinavia).

# Input data

As input the model takes two past satellite images of effective cloudiness, spaced one hour apart. It also needs forcings from a dynamic model (MEPS). The forcings are needed for the history part as well as for all future predictions. The model predicts future cloudiness in 1-hour intervals. The model is autoregressive and can produce forecasts of any length as long as the forcings are available. In practice the quality of the predictions degrades as a function lead time.

The model also utilizes static environmental information, located at https://lake.fmi.fi/cloudcast-v2/const/meps-const-v4.zarr (anonymous s3-access available). Input data needs to be in anemoi-dataset format.

# Inference

See the inference-directory in the code repo for details.

# Preprocessing

The geographical domain is that of MEPS (MEPS25D): northern europe in lambert conformal conic projection, 2.5 km grid. The satellite coverage is very poor in the north-east corner of the domain which can be seen as a visible saw blade-shaped static artifact.

The NWCSAF effective cloudiness has known issues which we try to correct **before** the data is fed to the neural network to make a prediction. Our training data set does not have this correction.

1. Cumulus clouds are reported as 100% cloud cover, probably due to the resolution of the data. We try to fix this by using short range radiation information to decrease the cloud cover
  * [https://github.com/fmidev/himan/blob/master/himan-scripts/nwcsaf-cumulus.lua](https://github.com/fmidev/himan/blob/master/himan-scripts/nwcsaf-cumulus.lua)

2. Shallow low level clouds are sometimes not detected during autumn/winter. We have different methods to try to fix this big quality issue
  * [https://github.com/fmidev/himan/blob/master/himan-scripts/nwcsaf-low-and-high-cloud.lua](https://github.com/fmidev/himan/blob/master/himan-scripts/nwcsaf-cumulus.lua)

3. Unnatural "clear sky" areas if there are clouds on different levels, probably due to shadow effect. We have methods to try to fix this too.

These corrections are made with Himan tool. Training data does not have these corrections.

Additionally in spring and autumn sometime at the sun flares hitting satellite sensor causes the cloudiness to increase to nearly 100% for the whole domain for some timesteps, usually at late evening.

# Technical details

The model is written with Torch and uses Lighting framework extensively. It also uses anemoi-datasets for input data. Training resolution is 553x475 pixels, meaning 5km physical resolution. In operations we downscale the output data to 2.5km. Training data contained 30 years of CERRA data in hourly intervals and 5 years of MEPS+NWCSAF data for fine-tuning.

## Verification


## Weights


## Running the model.

Use the provided Container file to build a container with all dependencies installed.

The see the inference-directory in the code repo for details on how to run inference with the model.

# Training
