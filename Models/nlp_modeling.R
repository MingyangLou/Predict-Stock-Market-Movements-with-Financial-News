#### Predict Stock Market Movements with Financial News

library(tm)
library(dplyr)
library(lubridate)
library(caTools)
library(caret)
library(rpart)
library(randomForest)
library(boot)
library(MASS)
library(reshape2)
library(ggplot2)


#-----1 prepare------

# load data
news <- read.csv('news.csv', stringsAsFactors = FALSE)
label <- read.csv('SP_500_label.csv')
tech_indicators <- read.csv('tech_indicator.csv')

news$datetime <- as.Date(news$datetime,'%Y-%m-%d')
label$label <- as.factor(label$label)
label$Date <- as.Date(label$Date,'%Y-%m-%d')
tech_indicators <- tech_indicators[,c('Date', 'MA_Cross', 'ROC_5', 'RSI')]
tech_indicators$Date <- as.Date(tech_indicators$Date, '%m/%d/%y')

# data preprocessing
corpus.title <- Corpus(VectorSource(news$headline))
corpus.text <- Corpus(VectorSource(news$text))
strwrap(corpus.title[[1]])
strwrap(corpus.text[[1]])

corpus.title <- tm_map(corpus.title, removePunctuation)
corpus.text <- tm_map(corpus.text, removePunctuation)

corpus.title <- tm_map(corpus.title, tolower) 
corpus.text <- tm_map(corpus.text, tolower)

corpus.title <- tm_map(corpus.title, removeWords, c(stopwords('english')))
corpus.text <- tm_map(corpus.text, removeWords, c(stopwords("english"), 'reuters'))

corpus.title <- tm_map(corpus.title, stemDocument)
corpus.text <- tm_map(corpus.text, stemDocument)

remove.number <-function(y){
  return(gsub("\\b\\d+\\b", "", y))
}
corpus.title <- tm_map(corpus.title, remove.number)
corpus.text <- tm_map(corpus.text, remove.number)

# bag of words
# word frequencies
frequencies.title <- DocumentTermMatrix(corpus.title)
frequencies.text <- DocumentTermMatrix(corpus.text)

sparse.title <- removeSparseTerms(frequencies.title, 0.99)
sparse.text <- removeSparseTerms(frequencies.text, 0.90)

news.wf.title <- as.data.frame(as.matrix(sparse.title))
news.wf.text <- as.data.frame(as.matrix(sparse.text))
news.wf <- cbind(news.wf.title, news.wf.text)
names(news.wf) <- make.unique(names(news.wf))
colnames(news.wf) <- make.names(colnames(news.wf))

news.wf$Date = news$datetime

# sum word frequencies on the same day
news.data <- aggregate(. ~ Date, news.wf, sum)

# add technical indicators
news.data <- merge(tech_indicators, news.data, by = 'Date')

# add labels
news.data <- merge(label, news.data, by = 'Date')


#----2 modeling-----

tableAccuracy <- function(test, pred) {
  t <- table(test, pred)
  acc <- sum(diag(t))/length(test)
  return(acc)
}

# split data
train <- news.data %>% filter(year(Date) <= 2011)
test <- news.data %>% filter(year(Date) >= 2012)
#val <- val_test[1:100,]
#test <- val_test[-(1:100),]


# Baseline
table(train$label)
table(test$label)
263/(209 + 263) # 0.5572034


# Logistic Regression
mod.log <- glm(label ~ .-Date, data = train, family = 'binomial')
preds.log <- predict(mod.log, newdata = test, type = 'response')
table(test$label, preds.log > 0.5)
tableAccuracy(test$label, preds.log > 0.5) # 0.4957627


# LDA
mod.lda <- lda(label ~ .-Date, data = train)
preds.lda <- predict(mod.lda, newdata = test)$class
table(test$label, preds.lda)
tableAccuracy(test$label, preds.lda) # 0.4745763


# CART
set.seed(123)
train.cart <- train(label ~ . -Date,
                    data = train,
                    method = 'rpart',
                    tuneGrid = data.frame(cp = seq(0, 0.1, 0.002)),
                    trControl = trainControl(method = 'cv', number = 5),
                    metric = 'Accuracy')

ggplot(train.cart$results, aes(x = cp, y = Accuracy)) +
  geom_point() + geom_line() +
  xlab("Complexity Parameter (cp)")

train.cart$bestTune
mod.cart <- train.cart$finalModel
prp(mod.cart) # same as baseline

preds.cart <- predict(mod.cart, newdata = test, type = 'class')  
table(test$label, preds.cart)
tableAccuracy(test$label, preds.cart) # 0.5572034


# Random Forest
set.seed(123)
mod.naiverf <- randomForest(label ~ . -Date, data = train)
preds.naiverf <- predict(mod.naiverf, newdata = test)
table(test$label, preds.naiverf)
tableAccuracy(test$label, preds.naiverf) # 0.5614407

set.seed(345)
train.rf <- train(label ~ . -Date,
                  data = train,
                  method = 'rf',
                  tuneGrid = data.frame(mtry = 1:35),
                  trControl = trainControl(method = 'cv', number = 5, verboseIter = TRUE),
                  metric = 'Accuracy')

ggplot(train.rf$results, aes(x = mtry, y = Accuracy)) +
  geom_point() + geom_line()

train.rf$bestTune
mod.rf <- train.rf$finalModel

preds.rf <- predict(mod.rf, newdata = test)
table(test$label, preds.rf)
tableAccuracy(test$label, preds.rf) # 0.5572034


# Boosting
tGrid = expand.grid(n.trees = (1:50)*50, interaction.depth = c(1,2,4,6,8),
                    shrinkage = 0.01, n.minobsinnode = 10)

set.seed(123)
train.boost <- train(label ~. -Date,
                     data = train,
                     method = "gbm",
                     tuneGrid = tGrid,
                     trControl = trainControl(method = "cv", number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     distribution = "bernoulli")

ggplot(train.boost$results, aes(x = n.trees, y = Accuracy,
                                colour = as.factor(interaction.depth))) +
  geom_line() + scale_color_discrete(name = "interaction.depth")
train.boost$bestTune

mod.boost <- train.boost$finalModel
test.mm <- as.data.frame(model.matrix(label ~.-Date +0, data = test))

preds.boost <- predict(mod.boost, newdata = test.mm, n.trees = 200, type = "response")
table(test$label, preds.boost)
tableAccuracy(test$label, preds.boost < 0.5) # 0.5741525


# Neural Network
nn.result <- read.csv('Neural_Networks_pred.csv')
preds.nn <- nn.result$NN
table(test$label, preds.nn)
tableAccuracy(test$label, preds.nn)


# Time Series
ts.result <- read.csv('time_series.csv')
ts.result$Date <- as.Date(ts.result$Date, '%Y-%m-%d')
ts.result <- ts.result[ts.result$Date %in% val_test$Date,]

preds.ts <- ts.result$label
table(test$label, preds.ts)
tableAccuracy(test$label, preds.ts)


# Majority Vote
vote <- read.csv('test_vote_df.csv')
preds.vote <- vote$vote
table(test$label, preds.vote)
tableAccuracy(test$label, preds.vote)


# Blending
# val.preds.rf <- predict(mod.rf, newdata = val)
# val.preds.boost <- boostResult[1:100,]$prediction
# val.preds.nn <- nn.result[1:100,]$NN
# val.preds.log <- predict(mod.log, newdata = val, type = 'response')
# val.preds.log <- ifelse(val.preds.log > 0.5, 1, 0)
# val.preds.lda <- predict(mod.lda, newdata = val)$class
# val.preds.ts <- ts.result[1:100,]$label
# 
# val.blending.df <- data.frame(label = val$label,
#                               rf_preds = val.preds.rf,
#                               boost_preds = val.preds.boost,
#                               nn_preds = val.preds.nn,
#                               log_preds = val.preds.log,
#                               lda_preds = val.preds.lda,
#                               ts_preds = val.preds.ts)
# 
# mod.blend <- glm(label ~ . -1, data = val.blending.df, family = 'binomial')
# 
# preds.boost.class <- ifelse(preds.boost < 0.5, 1, 0)
# preds.log.class <- ifelse(preds.log > 0.5, 1, 0)
# test.blending.df <- data.frame(label = test$label,
#                                rf_preds = preds.rf,
#                                boost_preds = preds.boost.class,
#                                nn_preds = preds.nn,
#                                log_preds = preds.log.class,
#                                lda_preds = preds.lda,
#                                ts_preds = preds.ts)
# 
# test.preds.blend <- predict(mod.blend, newdata = test.blending.df, type = 'response')
# 
# table(test$label, test.preds.blend > 0.5)
# tableAccuracy(test$label, test.preds.blend > 0.5)



#----3 bootstrap-----

tableAccuracy <- function(label, pred) {
  t = table(label, pred)
  a = sum(diag(t))/length(label)
  return(a)
}

tableTPR <- function(label, pred) {
  t = table(label, pred)
  return(t[2,2]/(t[2,1] + t[2,2]))
}

tableFPR <- function(label, pred) {
  t = table(label, pred)
  return(t[1,2]/(t[1,1] + t[1,2]))
}

boot_accuracy <- function(data, index) {
  labels <- data$label[index]
  predictions <- data$prediction[index]
  return(tableAccuracy(labels, predictions))
}

boot_tpr <- function(data, index) {
  labels <- data$label[index]
  predictions <- data$prediction[index]
  return(tableTPR(labels, predictions))
}

boot_fpr <- function(data, index) {
  labels <- data$label[index]
  predictions <- data$prediction[index]
  return(tableFPR(labels, predictions))
}

boot_all_metrics <- function(data, index) {
  acc = boot_accuracy(data, index)
  tpr = boot_tpr(data, index)
  fpr = boot_fpr(data, index)
  return(c(acc, tpr, fpr))
}

B <- 100000

# Logistic Regression
df.logit <- data.frame(labels = test$label,
                       predictions = preds.log > 0.5)
set.seed(2345)
logit.boot <- boot(df.logit, boot_all_metrics, R = B)
boot.ci(logit.boot, index = 1, type = "basic")
boot.ci(logit.boot, index = 2, type = "basic")
boot.ci(logit.boot, index = 3, type = "basic")

# LDA
df.lda <- data.frame(labels = test$label,
                     predictions = preds.lda)
set.seed(3456)
lda.boot = boot(df.lda, boot_all_metrics, R = B)
boot.ci(lda.boot, index = 1, type = "basic")
boot.ci(lda.boot, index = 2, type = "basic")
boot.ci(lda.boot, index = 3, type = "basic")

# Naive Random Forest
df.naiverf <- data.frame(labels = test$label,
                         predictions = preds.naiverf)
set.seed(4567)
naiverf.boot <- boot(df.naiverf, boot_all_metrics, R = B)
boot.ci(naiverf.boot, index = 1, type = "basic")
boot.ci(naiverf.boot, index = 2, type = "basic")
boot.ci(naiverf.boot, index = 3, type = "basic")

# Random Forest (CV)
df.rf <- data.frame(labels = test$label,
                    predictions = preds.rf)
set.seed(5678)
rf.boot <- boot(df.rf, boot_all_metrics, R = B)
boot.ci(rf.boot, index = 1, type = "basic")
boot.ci(rf.boot, index = 2, type = "basic")
boot.ci(rf.boot, index = 3, type = "basic")

# Boosting
df.boost <- data.frame(labels = test$label,
                       predictions = preds.boost)
set.seed(123)
boost.boot <- boot(df.boost, boot_all_metrics, R = B)
boot.ci(boost.boot, index = 1, type = "basic")
boot.ci(boost.boot, index = 2, type = "basic")
boot.ci(boost.boot, index = 3, type = "basic")

# Neural Network
df.nn <- data.frame(labels = test$label,
                    predictions = preds.nn)
set.seed(7890)
nn.boot <- boot(df.nn, boot_all_metrics, R = B)
boot.ci(nn.boot, index = 1, type = "basic")
boot.ci(nn.boot, index = 2, type = "basic")
boot.ci(nn.boot, index = 3, type = "basic")

# Time Series
df.ts <- data.frame(labels = test$label,
                    predictions = preds.ts)
set.seed(8901)
ts.boot <- boot(df.ts, boot_all_metrics, R = B)
boot.ci(ts.boot, index = 1, type = "basic")
boot.ci(ts.boot, index = 2, type = "basic")
boot.ci(ts.boot, index = 3, type = "basic")

# Majority Vote
df.vote <- data.frame(labels = test$label,
                      predictions = preds.vote)
set.seed(9012)
vote.boot <- boot(df.vote, boot_all_metrics, R = B)
boot.ci(vote.boot, index = 1, type = "basic")
boot.ci(vote.boot, index = 2, type = "basic")
boot.ci(vote.boot, index = 3, type = "basic")