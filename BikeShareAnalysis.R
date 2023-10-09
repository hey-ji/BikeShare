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

library(tidymodels)

bike_training <- vroom("BikeShare/train.csv") %>%
  select(-casual, -registered)
bike_test <- vroom("BikeShare/test.csv")


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



# Random Forest -----------------------------------------------------------
install.packages("rpart")
install.packages('ranger')
library(tidymodels)
library(ranger)
library(vroom)
library(tidyverse)
library(tidymodels)

my_forest_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=250) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

## Create a workflow with model & recipe
#1. read in the data
setwd("/Users/student/Desktop/STAT348")
bike_training <- vroom("BikeShare/train.csv") %>%
  select(-casual, -registered)
bike_test <- vroom("BikeShare/test.csv")
bike_training$count <- log(bike_training$count) # change a response variable into log scale

# Create a recipe
my_recipe <- recipe(count~., data = bike_training) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>%
  step_mutate(weather=factor(weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>%
  step_rm(datetime) 

prepped_recipe <- prep(my_recipe)  # Sets up the preprocessing using myDataS
bake(prepped_recipe, new_data=bike_training)
bake(prepped_recipe, new_data=bike_test)

# Create a workflow
forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_forest_mod)

## Set up grid of tuning values
tuning_forest_grid <- grid_regular(mtry(range=c(1,8)),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

## Set up K-fold CV
folds <- vfold_cv(bike_training, v = 10, repeats=1)

CV_results <- forest_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_forest_grid,
            metrics=metric_set(rmse, mae))

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best("rmse")

final_wf <- forest_wf %>%
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
vroom_write(x=bike_predictions, file="BikeShare/bikepredsRandomForest.csv", delim=",")


# Stacking ----------------------------------------------------------------
install.packages('stacks')

library(vroom)
library(tidyverse)
library(tidymodels)
library(stacks)

## read in data
setwd("/Users/student/Desktop/STAT348")
bike_training <- vroom("BikeShare/train.csv") %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))
bike_test <- vroom("BikeShare/test.csv")

## create a recipe
my_recipe <- recipe(count~., data = bike_training) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>%
  step_mutate(weather=factor(weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())
            

prepped_recipe <- prep(my_recipe)  # Sets up the preprocessing using myDataS
bake(prepped_recipe, new_data=bike_training)
bake(prepped_recipe, new_data=bike_test)

# Split data for CV
folds <- vfold_cv(bike_training, v = 5, repeats=1)

## Create a control grid
untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a model

## Penalized regression model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow15
preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model)

## Grid of values to tune over
preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5) ## L^2 total tuning possibilities
## Run the CV1
preg_models <- preg_wf %>%
  tune_grid(resamples=folds,
          grid=preg_tuning_grid,
          metrics=metric_set(rmse),
          control = untunedModel) # including the control grid in the tuning ensures you can
# call on it later in the stacked model

## Create other resampling objects with different ML algorithms to include in a stacked model, for example

# linear model
lin_model <- linear_reg() %>% #Type of model3
  set_engine("lm") # Engine = What R function to use4

lin_reg_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(lin_model)

lin_reg_model <-
fit_resamples(
              lin_reg_workflow,
              resamples = folds,
              metrics = metric_set(rmse),
              control = tunedModel
)

# penalize regression
library(vroom)
library(tidyverse)
library(tidymodels)
library(poissonreg)

preg_model <- linear_reg(penalty=1, mixture=0.5) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) 

preg_model <-
  fit_resamples(
    preg_wf,
    resamples = folds,
    metrics = metric_set(rmse),
    control = tunedModel
  )

# random forest
my_forest_mod <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=250) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

tuning_forest_grid <- grid_regular(mtry(range=c(1,8)),
                                   min_n(),
                                   levels = 5) ## L^2 total tuning possibilities


forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_forest_mod)

forest_model <-
  tune_grid(
    forest_wf,
    resamples = folds,
    grid=tuning_forest_grid,
    metrics = metric_set(rmse),
    control = tunedModel
  )

# Specify with models to include
my_stack <- stacks() %>%
add_candidates(lin_reg_model) %>%
add_candidates(preg_models) %>%
add_candidates(forest_model) 


## Fit the stacked model
stack_mod <- my_stack %>%
blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset

## If you want to build your own metalearner you'll have to do so manually
## using
stackData <- as_tibble(my_stack)


## Formatting for submission
bike_predictions <- exp(predict(stack_mod, new_data=bike_test))
bike_predictions$datetime <- as.character(format(bike_test$datetime))
bike_predictions <- bike_predictions %>% 
  rename(count=.pred) %>%
  select(datetime, count)
vroom_write(x=bike_predictions, file="BikeShare/bikepredsStacking.csv", delim=",")


# Finishing up the Kaggle Bikeshare ---------------------------------------
library(tidymodels)
library(ranger)
library(vroom)
library(tidyverse)
library(tidymodels)
library(dplyr)

# set up a random forest
my_forest_mod <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

## Create a workflow with model & recipe
#1. read in the data
setwd("/Users/student/Desktop/STAT348")
bike_training <- vroom("BikeShare/train.csv") %>%
  select(-casual, -registered)
bike_test <- vroom("BikeShare/test.csv")
bike_training$count <- log(bike_training$count) # change a response variable into log scale

#2. create a recipe
my_recipe <- recipe(count~., data = bike_training) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>%
  step_mutate(weather=factor(weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_date(datetime, features = "year") %>%
  step_mutate(datetime_hour=factor(datetime_hour)) %>%
  step_mutate(datetime_year=factor(datetime_year)) %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(my_recipe)  # Sets up the preprocessing using myDataS
bake(prepped_recipe, new_data=bike_training)
bake(prepped_recipe, new_data=bike_test)


#3. Create a workflow
forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_forest_mod)

## Set up grid of tuning values
tuning_forest_grid <- grid_regular(mtry(range=c(1,10)),
                                   min_n(),
                                   levels = 5) ## L^2 total tuning possibilities

## Set up K-fold CV
folds <- vfold_cv(bike_training, v = 10, repeats=1)

CV_results <- forest_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_forest_grid,
            metrics=metric_set(rmse))

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best("rmse")

final_wf <- forest_wf %>%
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
vroom_write(x=bike_predictions, file="BikeShare/bikepredsLastSubmission2.csv", delim=",")
