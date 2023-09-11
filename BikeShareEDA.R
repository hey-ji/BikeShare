##
## Bike Share EDA Code
##

## Libraries
library(tidyverse)
library(vroom)
library(DataExplorer)
library(GGally)

## Read in the Data
setwd("/Users/student/Desktop/STAT348")
bike <- vroom("KaggleBikeShare/train.csv")
bike

## Perform an EDA and identify key features
dplyr::glimpse(bike) #lists the variable type of each column
DataExplorer::plot_intro(bike) #visualization of glimpse()
DataExplorer::plot_correlation(bike) #correlation heat map between variables
DataExplorer::plot_bar(bike) #bar charts of all discrete variables
DataExplorer::plot_histrograms(bike) #histograms of all numerical variables
DataExplorer::plot_missing(bike) #percent missing in each column
GGally::ggpairs(bike) #1/2 scatterplot and 1/2 correlation heat map

## Planning (picking up 4 variables)
# weather, season, atemp, windspeed

bike$weather <- factor(bike$weather, levels=c(1,2,3,4), labels=c("Clear", "Cloudy", "Light Precip", "Heavy Precip"))
bike$season <- factor(bike$season, levels=c(1,2,3,4), labels=c("spring","summer","fall","winter"))

#1. weather plot
ggplot(data=bike, mapping=aes(x=weather, y=count)) +
  geom_boxplot() +
  xlab("weather")

ggplot(data=bike, aes(x=count, color=weather)) + geom_histogram()

#2. season plot
ggplot(data=bike, mapping=aes(x=season, y=count)) +
  geom_boxplot() +
  xlab("season")

#3. atemp plot ("feels like" temperature in Celsius)
ggplot(data=bike, mapping=aes(x=atemp, y=count)) +
  geom_point() +
  geom_smooth(se=FALSE) +
  xlab("feels like temperature in Celsius")

#4. windspeed plot
ggplot(data=bike, mapping=aes(x=windspeed, y=count)) +
  geom_point() +
  geom_smooth(se=FALSE) +
  xlab("wind speed")
