# load necessary packages -----------------------------------------------------
library(tidymodels)
library(stacks)
library(tibble)
library(dplyr)




# "load" in commute data ;) ---------------------------------------------------
commute <- 
  tibble(
    mode = 
      sample(
        c("subaru", "john deere", "bicycle"), 
        500, 
        TRUE
      ),
    temp_f = rnorm(500, 50, 10),
    time_h = case_when(
      mode == "subaru" ~ rnorm(1, 29, 2),
      mode == "john deere" ~ rnorm(1, 32, 4),
      mode == "bicycle" ~ rnorm(1, 35, 3),
    )
  ) %>%
  rowwise() %>%
  mutate(
    time_h = time_h - rnorm(1, (temp_f - 50) / 10),
    food_c = case_when(
      mode == "bicycle" ~ rnorm(1, 1, .2),
      TRUE ~ rnorm(1, 0, .5)
    ),
    food_c = food_c + (time_h * rnorm(1, .3, .05))
  )



# define model specifications -------------------------------------------------
splits <- initial_split(commute)

commute_train <- training(splits)
commute_test <- testing(splits)

folds <- vfold_cv(commute_train, v = 5)

commute_rec <-
  recipe(food_c ~ ., commute_train) %>%
  step_dummy(all_nominal()) %>%
  step_zv(all_predictors())

lr_res <-
  fit_resamples(
    workflow() %>%
      add_recipe(commute_rec) %>%
      add_model(linear_reg() %>% set_engine("lm")),
    resamples = folds,
    control = control_stack_resamples(),
    metrics = metric_set(rmse)
  )

knn_res <- 
  tune_grid(
    workflow() %>%
      add_recipe(commute_rec) %>%
      add_model(
        nearest_neighbor(
          neighbors = tune()) %>% 
          set_engine("kknn") %>% 
          set_mode("regression")
      ),
    resamples = folds,
    grid = 4,
    control = control_stack_grid(),
    metrics = metric_set(rmse)
  )

nn_res <- 
  tune_grid(
    workflow() %>%
      add_recipe(commute_rec) %>%
      add_model(
        mlp(
          hidden_units = tune(), 
          dropout = tune()) %>% 
          set_engine("keras") %>% 
          set_mode("regression")
      ),
    resamples = folds,
    grid = 6,
    control = control_stack_grid(),
    metrics = metric_set(rmse)
  )



# check out the model specifications ------------------------------------------
lr_res

knn_res

nn_res



# build the ensemble ----------------------------------------------------------
st <- 
  stacks() %>%
  add_candidates(lr_res) %>%
  add_candidates(knn_res) %>%
  add_candidates(nn_res) %>%
  blend_predictions() %>%
  fit_members()

st

# check it out! ---------------------------------------------------------------
autoplot(st)

st_preds <- 
  predict(st, commute_test) %>%
  bind_cols(commute_test)

ggplot(st_preds) +
  aes(x = food_c, y = .pred) + 
  geom_point()

st_preds <- 
  predict(st, commute_test, members = TRUE) %>%
  bind_cols(commute_test)

st_preds %>%
  pivot_longer(
    cols = c(.pred, contains("_res")),
    names_to = "model",
    values_to = "prediction"
  ) %>%
  ggplot() +
  aes(
    x = food_c,
    y = prediction,
    col = model
  ) + 
  geom_point()


