
#####################################################################
#Author: Shahidul Islam, DrPH
#Code File for MovieLens project for Data Science Capstone Course
#11.29.2024
######################################################################


##########################################################
# Create edx and final_holdout_test sets (Code provided by the course materials)
##########################################################

# Note: this process could take a couple of minutes

# if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
# if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(sqldf)

setwd("C:/Users/ShahS/OneDrive/EDX.ORG/DataScienceCertificate/CAPSTONE")
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

## *******We created the 'edx' and 'final_holdout_test' (Validation) data 
#sets using the code above and saved them locally, as generating these data 
#sets can be time-consuming. By loading the data from the local drive, we were 
#able to significantly improve run time.********

###data out;
#save(edx, file="edx.Rdata")
#save(final_holdout_test, file="final_holdout_test.Rdata")
#load(file="edx.Rdata")
#load(file="final_holdout_test.Rdata")

##############assessment
#How many movie ratings are in each of the following genres in the edx dataset?
xx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Which movie has the greatest number of ratings?

edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# What are the five most given ratings in order from most to least?

edx %>% group_by(rating) %>% 
  summarize(count = n()) %>% 
  top_n(5) %>%
  arrange(desc(count))  


# you will train a machine learning algorithm using the inputs 
# in one subset to predict movie ratings in the validation set. 

###Data Exploration##################################

#To get a visual of of the dataset, we can print a few rows of the data, look at data dimension, variable type, etc.
head(edx, n=5)
names(edx)
str(edx)

##dataframe edx includes 9000055 obs. of  6 variables. Note that genres variable contain multiple genres seperated by pipe character
#[1] "userId"    "movieId"   "rating"    "timestamp" "title"     "genres"   

#69878 unique users have  9000055 observations in the dataset. 10677
sqldf("select count(distinct userId) from edx")
sqldf("select count( movieId) from edx")

#68534 unique users have 999999 observations
sqldf("select count(distinct userId) from final_holdout_test")

##Top 5 higest rated movies overall
edx %>%
  group_by(title) %>%
  summarize(N_ratings = n(), avg_ratings = mean(rating, na.rm = TRUE)) %>%
  arrange(desc(avg_ratings)) %>%
  slice_max(order_by = avg_ratings, n = 5)


#sqldf("select * from edx where title='Human Condition II, The (Ningen no joken II) (1959)'")

##Top 5 highest rated movies with >1000 user ratings
edx %>%
  group_by(title) %>%
  summarize(N_ratings = n(), avg_ratings = mean(rating, na.rm = TRUE)) %>%
  filter(N_ratings>1000)%>%
  arrange(desc(avg_ratings)) %>%
  slice_max(order_by = avg_ratings, n = 5)

##cleaning  the genres variable to parse values of genres--each movie is >=1 genres, let's ensure we get the count by each genres

edx %>% 
  separate_rows(genres, sep = "\\|") %>% 
  group_by(genres) %>% 
  summarize(N_ratings = n()) %>% 
  arrange(desc(N_ratings)) %>% 
  slice_max(order_by = N_ratings, n = 5)

# Review distribution of ratings
ggplot(edx, aes(x = rating)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Ratings", x = "Values", y = "Frequency") 


###evaluating outliers###

#movies with single user ratings

edx %>%
  group_by(movieId) %>%
  summarize(ratings = n()) %>%
  filter(ratings == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = mean(rating), n_rating = n()) %>%
  slice(1:5) %>%
  knitr::kable()


###looking at the distribution of users
edx %>% count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins =50 , color ="cyan") 
ggtitle("Distribution of Users who rated movies in original scale") 

###Distribution of users in a log scale to get better view of the distribution
edx %>% count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins =50 , color ="cyan") +
  scale_x_log10() +
  ggtitle("Distribution of Users who rated movies in log scale") 



# sqldf("select count(distinct userId) from edx")

# Histogram showing the distribution of average movie ratings
#given by users who have rated more than 1000 movies.

edx%>%group_by(userId)%>%
  filter(n()>1000)%>%
  summarize(mean_rating=mean(rating))%>%
  ggplot(aes(mean_rating))+
  geom_histogram(bins=40, color="cyan")+
  xlab("Average rating")+
  ylab("Count of users")+
  ggtitle("Mean ratings by # of users")



###############predictive models################################

# Square Root of mean of squared errors--Function to compute RMSE
RMSE <-  function(true_ratings, predicted_ratings){ 
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# 1. Naive model
mu<-mean(edx$rating)
mu

# RMSE for the naive model
validation<-final_holdout_test
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse

# create dataframe to store RMSE
rmse_results <- as.data.frame(tibble(Model ="Naive Average movie rating model", RMSE = naive_rmse))
rmse_results %>% knitr::kable()
rmse_results

#2. Movie effect Model
# estimating bias for movie ratings
movie_avg <- edx %>%
  group_by(movieId) %>% 
  summarize(bias = mean(rating - mu))

# examine distribution of bias
movie_avg %>% 
  ggplot(aes(bias)) +
  geom_histogram(bins = 50, color = "cyan") +
  ggtitle("Distribution of bias")

# Compute the movie based predicted ratings on validation dataset
rmse_movie_model <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu + bias) %>%
  pull(pred) #Extracts the pred column as a vector.

rmse_movieBased<- RMSE(validation$rating, rmse_movie_model)
rmse_results <- rmse_results %>% add_row(Model="Movie-Based Model", RMSE=rmse_movieBased)
rmse_results

###3. Movie+User effect model

user_avg <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(bias_user = mean(rating - mu - bias))

# Compute the predicted ratings on validation dataset (movie+user)
rmse_movie_user_model <- validation %>%
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  mutate(pred = mu+ bias + bias_user) %>%
  pull(pred)

rmse_movie_userbased <- RMSE(validation$rating, rmse_movie_user_model)
rmse_results <- rmse_results %>% add_row(Model="Movie+User Based Model", RMSE=rmse_movie_userbased)
rmse_results

### 4. Movie+user+Genre based model
genre_pop <- edx %>%
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  group_by(genres) %>%
  summarize(bias_user_genre = mean(rating - mu - bias - bias_user))

# Compute the predicted ratings on validation dataset

rmse_movie_user_genre_model <- validation %>%
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(genre_pop, by='genres') %>%
  mutate(pred = mu + bias + bias_user + bias_user_genre) %>%
  pull(pred)

rmse_movie_user_genre <- RMSE(validation$rating, rmse_movie_user_genre_model)
rmse_results <- rmse_results %>% add_row(Model="Movie+User+Genre Based Model", RMSE=rmse_movie_user_genre)
rmse_results


################Regularization models########################################################


###Regularization model--movie based###

lambdas <- seq(0, 10, 0.1)
# Compute the predicted ratings on validation dataset using different values of lambda
rmses <- sapply(lambdas, function(lambda) {
  
  # Calculate the average by movieId
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  #predicted ratings on validation data set
  predicted_ratings <- validation %>%
    left_join(b_i, by='movieId') %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  
  # Predict RMSE on validation set
  return(RMSE(validation$rating, predicted_ratings))
})

# lambda value that minimizes the RMSE
min_lambda <- lambdas[which.min(rmses)]
# Predict the RMSE on the validation set
rmse_regularized_movie <- min(rmses)
# Add the results to the results dataset
rmse_results <- rmse_results %>% add_row(Model="Regularized Movie-Based Model", RMSE=rmse_regularized_movie)
rmse_results

ggplot(data = data.frame(lambdas, rmses), aes(x = lambdas, y = rmses)) +
  geom_point(color = "blue") +
  ggtitle("RMSEs vs. Lambdas--MOvie based") +
  xlab("Lambdas") +
  ylab("RMSEs")


###Regularization Movie+user based model###

rmses <- sapply(lambdas, function(lambda) {
  # Calculate the average by user
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  b_u <- edx %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + lambda))
  
  # predicted ratings on validation dataset
  predicted_ratings <- validation %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  # Predict the RMSE on the validation set
  return(RMSE(validation$rating, predicted_ratings))
})

# lambda value that minimizes the RMSE
min_lambda <- lambdas[which.min(rmses)]

# Predict RMSE on validation set
rmse_regularized_movie_user <- min(rmses)
rmse_results <- rmse_results %>% add_row(Model="Regularized Movie+User Based Model", RMSE=rmse_regularized_movie_user)
rmse_results
ggplot(data = data.frame(lambdas, rmses), aes(x = lambdas, y = rmses)) +
  geom_point(color = "blue") +
  ggtitle("RMSEs vs. Lambdas--MOvie+user based") +
  xlab("Lambdas") +
  ylab("RMSEs")


###MOvie+user+genre based model###


# Compute the predicted ratings on validation dataset using different values of lambda
rmses <- sapply(lambdas, function(lambda) {
  # Calculate the average by user
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  # Calculate the average by user
  b_u <- edx %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + lambda))
  
  b_u_g <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_u_g = sum(rating - b_i - mu - b_u) / (n() + lambda))
  
  #predicted ratings on validation dataset
  predicted_ratings <- validation %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_u_g, by='genres') %>%
    mutate(pred = mu + b_i + b_u + b_u_g) %>%
    pull(pred)
  # Predict RMSE on validation set
  return(RMSE(validation$rating, predicted_ratings))
})

# lambda value that minimizes the RMSE
min_lambda <- lambdas[which.min(rmses)]
# Predict RMSE on validation set
rmse_regularized_movie_user_genre <- min(rmses)

rmse_results <- rmse_results %>% add_row(Model="Regularized Movie+User+Genre Based Model", RMSE=rmse_regularized_movie_user_genre)
rmse_results

ggplot(data = data.frame(lambdas, rmses), aes(x = lambdas, y = rmses)) +
  geom_point(color = "blue") +
  ggtitle("RMSEs vs. Lambdas--Movie+user+genre based") +
  xlab("Lambdas") +
  ylab("RMSEs")

