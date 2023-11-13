# Load required libraries
library(tidymodels)
library(vroom)
library(tidyverse)
library(embed)
library(discrim)
library(keras)

# Load training and test data
GGG_train <- vroom('C:/BYU/2023(5) Fall/STAT 348/Gouls-Goblins-and-Ghosts/train.csv',
                   show_col_types = FALSE)
GGG_test <- vroom('C:/BYU/2023(5) Fall/STAT 348/Gouls-Goblins-and-Ghosts/test.csv',
                  show_col_types = FALSE)

# Create a recipe for modeling
nn_recipe <- recipe(type ~ ., data = GGG_train) %>%
  update_role(id, new_role = "id") %>%
  step_lencode_glm(color, outcome = vars(type)) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

# Prepare the recipe
prepped_recipe <- prep(nn_recipe)

# Apply the recipe to the training data
baked <- bake(prepped_recipe, new_data = GGG_train)

# Create a Naive Bayes model
nn_model <- mlp(hidden_units = tune (),
                epochs = 50) %>%
  set_engine("nnet") %>%
  set_mode("classification")

# Create a workflow with the recipe and model
nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

# Define a tuning grid for hyperparameter optimization
tuning_grid <- grid_regular(hidden_units(range = c(1, 200)), levels = 10)

# Create cross-validation folds
folds <- vfold_cv(GGG_train, v = 5, repeats = 1)

# Tune the Naive Bayes model using cross-validation
tuned_nn <- nn_wf %>%
  tune_grid(resamples = folds, grid = tuning_grid)

# Show the best hyperparameter values based on accuracy
show_best(tuned_nn, metric = 'accuracy')

tuned_nn %>% collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x = hidden_units, y = mean)) + geom_line()

# Select the best hyperparameter values
bestTune <- tuned_nn %>%
  select_best("accuracy")

# Finalize the workflow with the best hyperparameter values
final_nn_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_train)

# Make predictions on the test data
GGG_predictions <- predict(final_nn_wf, new_data = GGG_test, type = "class") %>%
  bind_cols(GGG_test) %>%
  mutate(type = .pred_class) %>%
  select(id, type)

# Write the predictions to a CSV file
vroom_write(x = GGG_predictions, file = "C:/BYU/2023(5) Fall/STAT 348/Gouls-Goblins-and-Ghosts/GGG_preds.csv", delim = ",")
