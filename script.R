library(keras)
library(dplyr)
library(tidyr)
library(ggplot2)
library(skimr)

logit <- function(p) log(p) - log(1 - p)
logistic <- function(x) 1/(1 + exp(-x))

n <- 100000
set.seed(19880923)
df <- data_frame(x_1 = 6*runif(n) - 3) %>%
  mutate(y_1 = rbinom(n, 1, prob = logistic(-1 + 2 * x_1)),
         y_2 = rbinom(n, 1, prob = logistic(-1 + 2 * tanh(-1 + 2 * x_1))))

skimr::skim(df) %>% skim_print %>% with(numeric)  %>% mutate_if(is.numeric, round, 2) %>% DT::datatable()

df %>% 
  gather(y_id, y_val, y_1, y_2) %>%
  mutate(x_1_cat = cut_number(x_1, n = 50)) %>%
  group_by(x_1_cat, y_id) %>%
  summarise(p = mean(y_val),
            n = n()) %>%
  mutate(logit_p = logit(p)) %>%
  gather(transformacao, p, p, logit_p) %>%
  ggplot() +
  geom_point(aes(x = x_1_cat, y = p, colour = y_id)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
  facet_wrap(~forcats::fct_inorder(transformacao), nrow = 2, scales = "free_y") +
  labs(x = "x_1", colour = "resposta")

# modelo glm 1 ------------------------------------------------------
modelo_lm_1 <- glm(y_1 ~ x_1, data = df, family = binomial)

# coefficients
coef(modelo_lm_1)

# accuracy
conf_matrix_lm_1 <- table(modelo_lm_1$fitted.values > 0.5, df$y_1)
sum(diag(conf_matrix_lm_1))/sum(conf_matrix_lm_1)

# modelo keras 1 -------------------------------------------------------
input_keras_1 <- layer_input(1, name = "modelo_keras_1")

output_keras_1 <- input_keras_1 %>% 
  layer_dense(units = 1, name = "camada_unica") %>%
  layer_activation("sigmoid", input_shape = 1, name = "link_logit")

modelo_keras_1 <- keras_model(input_keras_1, output_keras_1)

summary(modelo_keras_1)

modelo_keras_1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr = 0.1),
  metrics = c('accuracy')
)

modelo_keras_1_fit <- modelo_keras_1 %>% fit(
  x = df$x_1, 
  y = df$y_1, 
  epochs = 2, 
  batch_size = 10000
)

# coefficients
modelo_keras_1 %>% get_layer("camada_unica") %>% get_weights

# accuracy
loss_and_metrics_1 <- modelo_keras_1 %>% evaluate(df$x_1, df$y_1, batch = 100000)
loss_and_metrics_1[[2]]




##############################################################



# modelo keras 2 -------------------------------------------------------
input_keras_2 <- layer_input(1, name = "modelo_keras_2")

output_keras_2 <- input_keras_2 %>%
  layer_dense(units = 1, name = "camada_um") %>% 
  layer_activation("tanh", input_shape = 1, name = "tanh_de_dentro") %>%
  layer_dense(units = 1, input_shape = 1, name = "camada_dois") %>% 
  layer_activation("sigmoid", input_shape = 1, name = "link_logit_final")

modelo_keras_2 <- keras_model(input_keras_2, output_keras_2)

summary(modelo_keras_2)

modelo_keras_2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr = 0.1),
  metrics = c('accuracy')
)

modelo_keras_2_fit <- modelo_keras_2 %>% fit(
  x = df$x_1, 
  y = df$y_2, 
  epochs = 4, 
  batch_size = 100
)

# coefficients
modelo_keras_2 %>% get_layer("camada_um") %>% get_weights
modelo_keras_2 %>% get_layer("camada_dois") %>% get_weights

# accuracy
loss_and_metrics_2 <- modelo_keras_2 %>% evaluate(df$x_1, df$y_2, batch_size = 100000)
loss_and_metrics_2[[2]]


# modelo glm 2 ------------------------------------------------------
modelo_lm_2 <- glm(y_2 ~ I(tanh(-1 + 1 * x_1)), data = df, family = binomial)

# coefficients
coef(modelo_lm_2)

# accuracy
conf_matrix_lm_2 <- table(modelo_lm_2$fitted.values > 0.5, df$y_2)
sum(diag(conf_matrix_lm_2))/sum(conf_matrix_lm_2)




x_1_e_uns <- matrix(c(df$x_1, rep(1, nrow(df))), ncol = 2)
betas_camada_um <- (modelo_keras_2 %>% get_layer("camada_um") %>% get_weights %>% unlist)
df$x_3 <- tanh(x_1_e_uns %*% betas_camada_um)

# modelo glm 3 ------------------------------------------------------
modelo_lm_3 <- glm(y_2 ~ x_3, data = df, family = binomial)

# coefficients
coef(modelo_lm_3)

# accuracy
conf_matrix_lm_3 <- table(modelo_lm_3$fitted.values > 0.5, df$y_2)
sum(diag(conf_matrix_lm_3))/sum(conf_matrix_lm_3)
