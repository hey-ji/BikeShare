library(vroom)
library(tidyverse)
library(tidymodels)

# read in data
setwd("/Users/student/Desktop/STAT348")
bike <- vroom("BikeShare/train.csv")
bike

# cleaning
bike$weather <- factor(bike$weather, levels=c(1,2,3,4), labels=c("Clear", "Cloudy", "Light Precip", "Heavy Precip"))

bike %>%
  filter(weather != "Heavy Precip")

#bike <- bike %>% 
#  mutate(weather = ifelse(weather=="Heavy Precip", "Light Precip", weather)
View(bike)




## engineering
# change to factors 
# create time of day variable

my_recipe <- recipe(count~., data = bike) %>%
  step_time(datetime, features = "hour") %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter"))

prepped_recipe <- prep(my_recipe)  # Sets up the preprocessing using myDataS
bake(prepped_recipe, new_data=bike)

