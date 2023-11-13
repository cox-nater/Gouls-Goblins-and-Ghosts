# Load required libraries
library(tidymodels)
library(vroom)
library(tidyverse)
library(bonsai)
library(lightgbm)
library(embed)
library(discrim)
library(doParallel)

cl <- makePSOCKcluster(5)
registerDoParallel(cl)

# Load training and test data
GGG_train <- vroom('C:/BYU/2023(5) Fall/STAT 348/Gouls-Goblins-and-Ghosts/train.csv',
                   show_col_types = FALSE)
GGG_test <- vroom('C:/BYU/2023(5) Fall/STAT 348/Gouls-Goblins-and-Ghosts/test.csv',
                  show_col_types = FALSE)

# Create a recipe for modeling
my_recipe <- recipe(type ~ ., data = GGG_train) %>%
  step_rm(id) %>%
  step_lencode_glm(color, outcome = vars(type))

# Prepare the recipe
prepped_recipe <- prep(my_recipe)

# Apply the recipe to the training data
baked <- bake(prepped_recipe, new_data = GGG_train)

# Create a boosted model
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

# Create a BART model
# BART_model <- bart(trees = tune()) %>% # BART figures out depth and learn_rate
#   set_engine("dbarts") %>% # might need to install
#   set_mode("classification")

# Create a workflow with the recipe and model
boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

# BART_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(BART_model)

# Define a tuning grid for hyperparameter optimization
tuning_gridA <- grid_regular(tree_depth(), trees(), learn_rate(), levels = 3)
#tuning_gridB <- grid_regular(trees(), levels = 3)

# Create cross-validation folds
folds <- vfold_cv(GGG_train, v = 5, repeats = 1)

# Tune the Boost model using cross-validation
CV_resultsA <- boost_wf %>%
  tune_grid(resamples = folds, grid = tuning_gridA)

# Tune the BART model using cross-validation
# CV_resultsB <- BART_wf %>%
#   tune_grid(resamples = folds, grid = tuning_gridB)

# Show the best hyperparameter values based on accuracy
show_best(CV_resultsA, metric = 'accuracy')
#show_best(CV_resultsB, metric = 'accuracy')

# Select the best hyperparameter values
bestTuneA <- CV_resultsA %>%
  select_best("accuracy")

# bestTuneB <- CV_resultsB %>%
#   select_best("accuracy")

# Finalize the workflow with the best hyperparameter values
final_boost_wf <- boost_wf %>%
  finalize_workflow(bestTuneA) %>%
  fit(data = GGG_train)

# final_BART_wf <- BART_wf %>%
#   finalize_workflow(bestTuneB) %>%
#   fit(data = GGG_train)

# Make predictions on the test data
GGG_predictionsA <- predict(final_boost_wf, new_data = GGG_test, type = "class") %>%
  bind_cols(GGG_test) %>%
  mutate(type = .pred_class) %>%
  select(id, type)

# GGG_predictionsB <- predict(final_BART_wf, new_data = GGG_test, type = "class") %>%
#   bind_cols(GGG_test) %>%
#   mutate(type = .pred_class) %>%
#   select(id, type)

# Write the predictions to a CSV file
vroom_write(x = GGG_predictionsA, file = "C:/BYU/2023(5) Fall/STAT 348/Gouls-Goblins-and-Ghosts/GGG_preds_boost.csv", delim = ",")
#vroom_write(x = GGG_predictionsB, file = "C:/BYU/2023(5) Fall/STAT 348/Gouls-Goblins-and-Ghosts/GGG_preds_BART.csv", delim = ",")


stopCluster(cl)
