library(tidymodels)  
library(readr)       # for importing data
library(vip)         # for variable importance plots



hotels <- 
  read_csv('https://tidymodels.org/start/case-study/hotels.csv') %>%
  mutate(across(where(is.character), as.factor))


hotels$country=as.character(hotels$country)
hotels$country=with(hotels,ifelse(country %in% c('PRT','GBR','FRA','ESP','DEU'),country,'otherCountry'))
hotels$country=as.factor(hotels$country)

set.seed(123)
splits      <- initial_split(hotels, strata = children)

hotel_other <- training(splits)
hotel_test  <- testing(splits)

set.seed(234)

val_set <- vfold_cv(hotel_other, 
                       strata = children,
                       v = 10)


#############  logistic regression with lasso  #############

lr_mod <- 
  logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")


holidays <- c("AllSouls", "AshWednesday", "ChristmasEve", "Easter", 
              "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")

lr_recipe <- 
  recipe(children ~ ., data = hotel_other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date, holidays = holidays) %>% 
  step_rm(arrival_date) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())


lr_workflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_recipe)


lr_reg_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 30))


lr_res <- 
  lr_workflow %>% 
  tune_grid(val_set,
            grid = lr_reg_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))


lr_plot <- 
  lr_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

lr_plot 


top_models <-
  lr_res %>% 
  show_best("roc_auc", n = 15) %>% 
  arrange(penalty) 
top_models



lr_best <- 
  lr_res %>% 
  select_best(metric = "roc_auc")
lr_best


  
lr_auc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Logistic Regression")

autoplot(lr_auc)


cores <- parallel::detectCores()
cores


#############  random forest ranger  #############


rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = tune()) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

rf_recipe <- 
  recipe(children ~ ., data = hotel_other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date) %>% 
  step_rm(arrival_date) 

rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)



rf_param <- extract_parameter_set_dials(rf_mod)

rf_param <- 
  rf_param %>% 
  finalize(x = hotel_other %>% select(-children))


rf_grid <- grid_regular(rf_param, levels = 2)

rf_res <- 
  rf_workflow %>% 
  tune_grid(val_set,
            grid = rf_grid,
            control = control_grid(save_pred = TRUE,verbose = TRUE),
            metrics = metric_set(roc_auc))


rf_res %>% 
  show_best(metric = "roc_auc")


autoplot(rf_res)


rf_best <- 
  rf_res %>% 
  select_best(metric = "roc_auc")
rf_best


rf_res %>% 
  collect_predictions()


rf_auc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Random forest ranger")

#############  random forest randomForest  #############


rf_randomForest_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("randomForest", num.threads = cores) %>% 
  set_mode("classification")

rf_randomForest_workflow <- 
  workflow() %>% 
  add_model(rf_randomForest_mod) %>% 
  add_recipe(rf_recipe)


rf_randomForest_param <- extract_parameter_set_dials(rf_randomForest_mod)

rf_randomForest_param <- 
  rf_randomForest_param %>% 
  finalize(x = hotel_other %>% select(-children))


rf_randomForest_grid <- grid_random(rf_randomForest_param,size = 10)

rf_randomForest_initial <- 
  rf_randomForest_workflow %>% 
  tune_grid(val_set,
            grid = rf_randomForest_grid,
            control = control_grid(save_pred = TRUE,verbose = TRUE),
            metrics = metric_set(roc_auc))


ctrl <- control_bayes(save_pred = TRUE,
                      verbose = TRUE,
                      parallel_over = 'resamples')


rf_randomForest_res <- 
  rf_randomForest_workflow %>% 
  tune_bayes(resamples = val_set,
             metrics = metric_set(roc_auc),
             initial = rf_randomForest_initial,
             param_info = rf_randomForest_param,
             iter = 10,
             control = ctrl)
    
  
rf_randomForest_res %>% 
  show_best(metric = "roc_auc")



autoplot(rf_randomForest_res)


rf_randomForest_best <- 
  rf_randomForest_res %>% 
  select_best(metric = "roc_auc")
rf_randomForest_best


rf_randomForest_res %>% 
  collect_predictions()


rf_randomForest_auc <- 
  rf_randomForest_res %>% 
  collect_predictions(parameters = rf_randomForest_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Random forest randomForest")



#############  decision tree rpart  #############


dt_mod <- 
  decision_tree(tree_depth = tune(), 
                min_n = tune(), 
                cost_complexity = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

dt_workflow <- 
  workflow() %>% 
  add_model(dt_mod) %>% 
  add_recipe(rf_recipe)




dt_param <- extract_parameter_set_dials(dt_mod)

dt_grid <- grid_random(dt_param,size = 10)

dt_initial <- 
  dt_workflow %>% 
  tune_grid(val_set,
            grid = dt_grid,
            control = control_grid(save_pred = TRUE,verbose = TRUE),
            metrics = metric_set(roc_auc))


ctrl <- control_bayes(save_pred = TRUE,
                      verbose = TRUE,
                      parallel_over = 'resamples')


dt_res <- 
  dt_workflow %>% 
  tune_bayes(resamples = val_set,
             metrics = metric_set(roc_auc),
             initial = dt_initial,
             param_info = dt_param,
             iter = 10,
             control = ctrl)

dt_res %>% 
  show_best(metric = "roc_auc")



autoplot(dt_res)


dt_best <- 
  dt_res %>% 
  select_best(metric = "roc_auc")
dt_best


dt_res %>% 
  collect_predictions()


dt_auc <- 
  dt_res %>% 
  collect_predictions(parameters = dt_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Decision tree")


###### podsumowanie

bind_rows(rf_auc, lr_auc, rf_randomForest_auc,dt_auc) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) + 
  coord_equal() + 
  scale_color_viridis_d(option = "plasma", end = .6)







last_rf_mod <- 
  rand_forest(mtry = 7, min_n = 3, trees = 1000) %>% 
  set_engine("randomForest", num.threads = cores) %>% 
  set_mode("classification")

# the last workflow
last_rf_workflow <- 
  rf_workflow %>% 
  update_model(last_rf_mod)

# the last fit
set.seed(345)
last_rf_fit <- 
  last_rf_workflow %>% 
  last_fit(splits)

last_rf_fit



last_rf_fit %>% 
  collect_metrics()


last_rf_fit %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 20)
