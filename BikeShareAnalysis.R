library(vroom)
library(tidyverse)
library(tidymodels)


# Read in the data --------------------------------------------------------

setwd("/Users/student/Desktop/STAT348")
bike_training <- vroom("BikeShare/train.csv") %>%
  select(-casual, -registered)
bike_test <- vroom("BikeShare/test.csv")

# Data Cleaning -----------------------------------------------------------

# bike_training$weather <- factor(bike_training$weather, levels=c(1,2,3,4), labels=c("Clear", "Cloudy", "Light Precip", "Heavy Precip"))
# 
# bike_training %>%
#   filter(weather != "Heavy Precip")
# 
# bike_test %>%
#   filter(weather != "Heavy Precip")
# 
#bike <- bike %>% 
#  mutate(weather = ifelse(weather=="Heavy Precip", "Light Precip", weather)

# Data Engineering --------------------------------------------------------

# change to factors 
# create time of day variable

my_recipe <- recipe(count~., data = bike_training) %>%
  step_mutate(weather=factor(weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter"))

prepped_recipe <- prep(my_recipe)  # Sets up the preprocessing using myDataS
bake(prepped_recipe, new_data=bike_training)
bake(prepped_recipe, new_data=bike_test)

# Linear Regression -------------------------------------------------------

my_mod <- linear_reg() %>% #Type of model3
  set_engine("lm") # Engine = What R function to use4

bike_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod) %>%
fit(data = bike_training) # Fit the workflow

bike_predictions <- predict(bike_workflow,
                            new_data=bike_test) # Use fit to predict
bike_predictions$datetime <- as.character(format(bike_test$datetime))
bike_predictions <- bike_predictions %>% 
  rename(count=.pred) %>%
  select(datetime, count) %>%
  mutate(count=pmax(count, 0))
vroom_write(x=bike_predictions, file="BikeShare/bikepreds.csv", delim=",")


# Poisson Regression ------------------------------------------------------

install.packages('poissonreg')
library(poissonreg)

pois_mod <- poisson_reg() %>% # Type of model
  set_engine("glm") # GLM = generalized linear model

bike_pois_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pois_mod) %>%
  fit(data = bike_training) # Fit the workflow

bike_predictions <- predict(bike_pois_workflow, new_data=bike_test)
bike_predictions$datetime <- as.character(format(bike_test$datetime))
bike_predictions <- bike_predictions %>% 
  rename(count=.pred) %>%
  select(datetime, count)
vroom_write(x=bike_predictions, file="BikeShare/bikepredsPois.csv", delim=",")


# Penalized Regression on Log count ---------------------------------------
library(tidymodels)
library(poissonreg) #if you want to do penalized, poisson regression

## change a response variable into log scale
bike_training$count <- log(bike_training$count)

## Create a recipe
my_recipe <- recipe(count~., data = bike_training) %>%
  step_mutate(weather=factor(weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=1



## Penalized regression model
preg_model <- linear_reg(penalty=1, mixture=0.5) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R
preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model) %>%
fit(data=bike_training)
exp(predict(preg_wf, new_data=bike_test))


bike_predictions <- exp(predict(preg_wf, new_data=bike_test))
bike_predictions$datetime <- as.character(format(bike_test$datetime))
bike_predictions <- bike_predictions %>% 
  rename(count=.pred) %>%
  select(datetime, count)
vroom_write(x=bike_predictions, file="BikeShare/bikepredsPenalizedRegression.csv", delim=",")


# Tuning models -----------------------------------------------------------

library(tidymodels)
library(poissonreg) #if you want to do penalized, poisson regression

## Penalized regression model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning6
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities
## Split data for CV
folds <- vfold_cv(bike_training, v = 10, repeats=1)

## Run the CV
CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")

## Finalize the Workflow & fit it
final_wf <-
preg_wf %>%
finalize_workflow(bestTune) %>%
fit(data=bike_training)

## Predict
final_wf %>%
predict(new_data = bike_test)

## Formatting for submission
bike_predictions <- exp(predict(final_wf, new_data=bike_test))
bike_predictions$datetime <- as.character(format(bike_test$datetime))
bike_predictions <- bike_predictions %>% 
  rename(count=.pred) %>%
  select(datetime, count)
vroom_write(x=bike_predictions, file="BikeShare/bikepredsTuningModel.csv", delim=",")


# Regression Tree ---------------------------------------------------------

install.packages("rpart")
library(tidymodels)

my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use
  set_mode("regression")

## Create a workflow with model & recipe
#1. my model & recipe

# change a response variable into log scale
bike_training$count <- log(bike_training$count)

# Create a recipe
my_recipe <- recipe(count~., data = bike_training) %>%
  step_mutate(weather=factor(weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=1

#2. create a workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

## Set up grid of tuning values
tuning_grid <- grid_regular(tree_depth(),
                            min_n(),
                            cost_complexity(),
                            levels = 5) ## L^2 total tuning possibilities
## Set up K-fold CV
folds <- vfold_cv(bike_training, v = 10, repeats=1)

CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae))

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best("rmse")

## Finalize the Workflow & fit it
final_wf <-
  preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=bike_training)

## Finalize workflow and predict
final_wf %>%
  predict(new_data = bike_test)

## Formatting for submission
bike_predictions <- exp(predict(final_wf, new_data=bike_test))
bike_predictions$datetime <- as.character(format(bike_test$datetime))
bike_predictions <- bike_predictions %>% 
  rename(count=.pred) %>%
  select(datetime, count)
vroom_write(x=bike_predictions, file="BikeShare/bikepredsRegressionTree.csv", delim=",")

