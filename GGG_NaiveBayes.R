# Load required libraries
library(tidymodels)
library(vroom)
library(tidyverse)
library(embed)
library(naivebayes)
library(discrim)

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

# Create a Naive Bayes model
nb_model <- naive_Bayes(Laplace = tune (), smoothness = tune ()) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

# Create a workflow with the recipe and model
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# Define a tuning grid for hyperparameter optimization
tuning_grid <- grid_regular(Laplace(), smoothness(), levels = 3)

# Create cross-validation folds
folds <- vfold_cv(GGG_train, v = 5, repeats = 1)

# Tune the Naive Bayes model using cross-validation
CV_results <- nb_wf %>%
  tune_grid(resamples = folds, grid = tuning_grid)

# Show the best hyperparameter values based on accuracy
show_best(CV_results, metric = 'accuracy')

# Select the best hyperparameter values
bestTune <- CV_results %>%
  select_best("accuracy")

# Finalize the workflow with the best hyperparameter values
final_nb_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_train)

# Make predictions on the test data
GGG_predictions <- predict(final_nb_wf, new_data = GGG_test, type = "class") %>%
  bind_cols(GGG_test) %>%
  mutate(type = .pred_class) %>%
  select(id, type)

# Write the predictions to a CSV file
vroom_write(x = GGG_predictions, file = "C:/BYU/2023(5) Fall/STAT 348/Gouls-Goblins-and-Ghosts/GGG_preds.csv", delim = ",")