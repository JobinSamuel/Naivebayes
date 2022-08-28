####Question -1
library(e1071)
library(gmodels)
library(scales)
library(tm)
#Read data
test <- read.csv("/Users/jobinsamuel/Desktop/Assignments/Naive bias/Datasets_Naive Bayes/SalaryData_Test.csv")
train <- read.csv("/Users/jobinsamuel/Desktop/Assignments/Naive bias/Datasets_Naive Bayes/SalaryData_Train.csv")
#Building naiveBayes Model
model <- naiveBayes(Salary~.,data = train) #Training the Model
pred <- predict(model,test) #Testing the model
table(pred)
table(test$Salary)
prop.table(table(pred)) #Produces proportion table
#Confusion Matrix
table(test$Salary,pred)
#Another way for showing the accuracy
acc <-percent((10550+1789)/15060)
acc           #Got an accuracy of 82%
#Since 82% of accuracy is not good enough so using laplace smoothing technique
##Laplace Smoothing
lap <- naiveBayes(train, train$Salary,laplace = .5)# We can change this laplace value
lap

#Evaluating model performance after applying laplace smoothing
test_pred_lap <- predict(lap,test)


#Crosstable of laplace smoothing model
CrossTable(test_pred_lap,test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))


## test accuracy
test_acc <- mean(test_pred_lap == test$Salary)
test_acc
#Here the accuracy has been increased to 98% 
acc <- percent((11211+3613)/15060)
acc

####Question -2
#Read data
sn <-read.csv("/Users/jobinsamuel/Desktop/Assignments/Naive bias/Datasets_Naive Bayes/NB_Car_Ad.csv")
#Spliting data into two train and test
train_sn <- sn[151:400 ,  ]
test_sn  <- sn[1:150, ]
#Building naiveBayes Model
Model <- naiveBayes(Purchased~.,data = train_sn)#Training the Model
predic <- predict(Model,test_sn)#Testing the model
table(predic)

table(train_sn$Purchased)
table(test_sn$Purchased)
prop.table(table(predic))#Produces proportion table
#Confusion Matrix
table(test_sn$Purchased,predic)

accu <-percent((125+15)/150)
accu
## test accuracy
test_accu <- mean(predic == test_sn$Purchased)
test_accu #Here we are already getting an accuracy of 93.3%

####Question -3
Tw <- read.csv("/Users/jobinsamuel/Desktop/Assignments/Naive bias/Datasets_Naive Bayes/Disaster_tweets_NB.csv")
str(Tw)
Tw_f <- factor(Tw$target)
# examining the type variable more carefully
str(Tw_f)
table(Tw_f)
# proportion of target
prop.table(table(Tw_f))

# build a corpus using the text mining (tm) package
str(Tw_f)
Tw_corpus <- Corpus(VectorSource(Tw$text))

Tw_corpus <- tm_map(Tw_corpus, function(x) iconv(enc2utf8(x), sub='byte'))

# cleaning up the corpus using tm_map()
corp_clean <- tm_map(Tw_corpus, tolower)
corp_clean <- tm_map(corp_clean, removePunctuation)
corp_clean <- tm_map(corp_clean, removeNumbers)
corp_clean <- tm_map(corp_clean, removeWords, stopwords())
corp_clean <- tm_map(corp_clean, stripWhitespace)

# creating a document-term sparse matrix
Tw_dtm <- DocumentTermMatrix(corp_clean)
Tw_dtm
View(Tw_dtm[1:10, 1:30])

# To view DTM we need to convert it into matrix first
Tw_matrix <- as.matrix(Tw_dtm)
str(Tw_matrix)

View(Tw_matrix[1:10, 1:10])

colnames(Tw_dtm)[1:50]

# creating training and test datasets
train_tw <- Tw[1:6000, ]
test_tw  <- Tw[6001:7613, ]

Tw_corpus_train <- corp_clean[1:6000]
Tw_corpus_test  <- corp_clean[6001:7613]

Tw_dtm_train <- Tw_dtm[1:6100, ]
Tw_dtm_test  <- Tw_dtm[6101:7613, ]

# check that the proportion  is similar
prop.table(table(Tw$target))

prop.table(table(train_tw$target))
prop.table(table(test_tw$target))

# indicator features for frequent words
# dictionary of words which are used more than 9 times
Tw_dict <- findFreqTerms(Tw_dtm_train, 9)

Tw_train <- DocumentTermMatrix(Tw_corpus_train, list(dictionary = Tw_dict))
Tw_test  <- DocumentTermMatrix(Tw_corpus_test, list(dictionary = Tw_dict))

Tw_test_matrix <- as.matrix(Tw_test)
View(Tw_test_matrix[1:10,1:10])

# convert counts to a factor
# custom function: if a word is used more than 0 times then mention 1 else mention 0
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
# Margin = 2 is for columns
# Margin = 1 is for rows
Tw_train <- apply(Tw_train, MARGIN = 2, convert_counts)
Tw_test  <- apply(Tw_test, MARGIN = 2, convert_counts)

View(Tw_test[1:10,1:10])

## building naiveBayes classifier.
Tw_classifier <- naiveBayes(Tw_train, train_tw$target)
Tw_classifier

##  Evaluating model performance
Tw_test_pred <- predict(Tw_classifier, Tw_test)

table(Tw_test_pred)
prop.table(table(Tw_test_pred))
## Crosstable
library(gmodels)

CrossTable(Tw_test_pred, test_tw$target,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy
test_ac <- mean(Tw_test_pred == test_tw$target)
test_ac

# On Training Data
Tw_train_pred <- predict(Tw_classifier, Tw_train)
Tw_train_pred

# train accuracy
training_acc = mean(Tw_train_pred == train_tw$target)
training_acc


### laplace smoothing
Tw_lap <- naiveBayes(Tw_train, train_tw$target,laplace = 1)# We can change this laplace value

# Evaluating model performance after applying laplace smoothing
Tw_test_pred_lap <- predict(Tw_lap, Tw_test) 

## crosstable of laplace smoothing model
CrossTable(Tw_test_pred_lap, test_tw$target,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy after laplace 
test_accu_lap <- mean(Tw_test_pred_lap == test_tw$target)
test_accu_lap


# prediction on train data for laplace model
Tw_train_pred_lap <- predict(Tw_lap,Tw_train)
Tw_train_pred_lap

# train accuracy after laplace
train_acc_lap = mean(Tw_train_pred_lap == train_tw$target)
train_acc_lap

