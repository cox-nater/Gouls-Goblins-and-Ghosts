library(tidymodels)
library(vroom)
library(tidyverse)
library(embed)
library(rsample)

GGG_train <- vroom('C:/BYU/2023(5) Fall/STAT 348/Gouls-Goblins-and-Ghosts/train.csv',
                   show_col_types = FALSE) %>%
  select(2:7)
GGG_test <- vroom('C:/BYU/2023(5) Fall/STAT 348/Gouls-Goblins-and-Ghosts/test.csv',
                  show_col_types = FALSE) %>%
  select(2:6) 

GGG_misstrain <- vroom('C:/BYU/2023(5) Fall/STAT 348/Gouls-Goblins-and-Ghosts/trainWithMissingValues.csv',
                       show_col_types = FALSE) %>%
  select(2:7)

my_recipe <- recipe(type ~ ., data = GGG_misstrain) %>%
  step_mutate( color = as.factor(color))%>%
  #step_impute_mean(all_numeric_predictors())
  step_impute_knn(all_predictors(), impute_with = imp_vars(all_predictors()), neighbors = 10) %>%
  #step_dummy(all_nominal_predictors())%>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = GGG_misstrain)

rmse_vec(GGG_train[is.na(GGG_misstrain)], baked[is.na(GGG_misstrain)])
