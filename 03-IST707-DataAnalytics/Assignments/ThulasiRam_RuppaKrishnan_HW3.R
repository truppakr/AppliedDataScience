########################################
## NAME YOUR R Files
##
## This file is an introduction to R
## for SYR.
## 
## Week 1
##
## It looks at StoryTeller data
##
## Practice - 
##     1) REading in csv
##     2) Looking at the data frame
##     3) Libraries
##     4) Setting a WD
##     5) installing
##     6) check for missing values
##     7) visual EDA part 1
##     8) Look at data types (str)
##    
##     RENAME DATA FILES SO THEY HAVE NO SPECIAL CHAR
#####################################################
## DO THIS ONCE
##install.packages("ggplot2")
library(arules)
library(grid)
library(arulesViz)
library(dplyr)  
library(ggplot2)
library(tidyverse)
library(reshape2)
library(ggrepel)
library(scales)
library(RColorBrewer)

# Clear objects
rm(list=ls())
## Set your working director to the path were your code AND datafile is
setwd("C:/Users/rkrishnan/Documents/01 Personal/MS/IST 707/week3")


## Read in .csv data
## Reference:C:\Users\rkrishnan\Documents\01 Personal\MS\IST 707\week3
## https://stat.ethz.ch/R-manual/R-devel/library/utils/html/read.table.html
## The data file and the R code must be in the SAME folder or you must use a path
## The name must be identical.

filename="bankdata_csv_all.csv"
MyStoryData <- read.csv(filename, header = TRUE, na.strings = "NA")


## Look at the data as a data frame
(head(MyStoryData))
(str(MyStoryData))

## Check for missing values
Total <-sum(is.na(MyStoryData))
cat("The number of missing values in StoryTeller data is ", Total )

## To clean this data, we can look through the variables and make sure that the data for each variable is in
## the proper range.
## The data shows the *number of students* in each category.
## This value cannot be negative - so 0 is the min. We do not know the max, but we
## might be suspecious of very large numbers. 

## Let's check each numerical variable to see that it is >= 

for(varname in names(MyStoryData)){
  ## Only check numeric variables
  if(sapply(MyStoryData[varname], is.numeric)){
    cat("\n", varname, " is numeric\n")
    ## Get median
    (Themedian <- sapply(MyStoryData[varname],FUN=median))
    ##print(Themedian)
    ## check/replace if the values are <=0 
    MyStoryData[varname] <- replace(MyStoryData[varname], MyStoryData[varname] < 0, Themedian)
  }
  
}

(MyStoryData)

## EXPLORE!
## For all assignments, explore your data.

## Tables are great!
(table(MyStoryData$School))

## loops - make all the tables at once
for(i in 1:ncol(MyStoryData)){
  print(table(MyStoryData[i]))
}

(colnames(MyStoryData))
(head(MyStoryData))
(MyStoryData)

MyStoryData$age_Discrete <- cut(MyStoryData$age, breaks = c(0,25,35,45,55,65,Inf),labels=c("Teens","YoungAdults","Adults","MiddleAge","Old","Senior"))
table(MyStoryData$age_Discrete)
hist(MyStoryData$age)
plot(MyStoryData$age_Discrete)

#MyStoryData$income_Discrete <- cut(MyStoryData$income, breaks = c(0,10000,20000,30000,40000,50000,Inf),labels=c("Low","BelowAverage","Average","AboveAverage","High","VeryHigh"))
#table(MyStoryData$income_Discrete)

min_income <- min(MyStoryData$income)
max_income <- max(MyStoryData$income)
bins = 6
width=(max_income - min_income)/bins;
MyStoryData$income_Discrete = cut(MyStoryData$income, breaks=seq(min_income, max_income, width),labels=c("Low","BelowAverage","Average","AboveAverage","High","VeryHigh"))
table(MyStoryData$income_Discrete)

hist(MyStoryData$age)
plot(MyStoryData$income_Discrete)


### Convert numeric to nominal for "children"

MyStoryData$children=factor(MyStoryData$children)

MyStoryData.P <- MyStoryData[,c(-1,-2,-5)]
(MyStoryData.P)
str(MyStoryData.P)
## Now load the transformed data into the apriori algorithm 


myRules = apriori(MyStoryData.P, parameter = list(supp = 0.1, conf = 0.5, maxlen =6))


# Show the top 5 rules, but only 2 digits
#```{r}
options(digits=2)
inspect(myRules[1:5])

summary(myRules)
#```

#Sorting stuff out
#```{r}
myRules<-sort(myRules, by="confidence", decreasing=TRUE)
#```

# Plot visulizations
#```{r}
arulesViz::ruleExplorer(myRules)
arulesViz::plot(myRules,method="graph",engine = "interactive",shading=NA)
#```

table(MyStoryData.P$save_act,MyStoryData.P$current_act)
prop.table(with(MyStoryData.P, table(MyStoryData.P$save_act, MyStoryData.P$current_act)),1)
table(MyStoryData.P$married,MyStoryData.P$children)
table(MyStoryData.P$car,MyStoryData.P$mortgage)
table(MyStoryData.P$car)
table(MyStoryData.P$mortgage)
table(MyStoryData.P$mortgage,MyStoryData.P$married)

MyStoryData.P$married_children <- paste("Married=",MyStoryData.P$married ,";Children=", ifelse(as.integer(as.character(MyStoryData.P$children))>0,"YES","NO"))
MyStoryData.P$Account <- paste(ifelse(paste(MyStoryData.P$save_act, MyStoryData.P$current_act)=="NO NO","NO","YES"))

MyStoryData.P2 <- MyStoryData.P[,c(-2,-3,-4,-6,-7)]
MyStoryData.P2$married_children <- as.factor(MyStoryData.P2$married_children)
MyStoryData.P2$Account <- as.factor(MyStoryData.P2$Account)
(MyStoryData.P2)
myRules = apriori(MyStoryData.P2, parameter = list(supp = 0.1, conf = 0.5, maxlen =6),appearance = list(rhs = c("pep=NO")))

#Sorting stuff out
#```{r}
myRules<-sort(myRules, by="confidence", decreasing=TRUE)
#```


# Show the top 5 rules, but only 2 digits
#```{r}
options(digits=2)
inspect(myRules[1:5])

summary(myRules)
#```


# Plot visulizations
#```{r}
arulesViz::ruleExplorer(myRules)
arulesViz::plot(myRules,method="graph",engine = "interactive",shading=NA)
#```

myRules = apriori(MyStoryData.P2, parameter = list(supp = 0.1, conf = 0.5, maxlen =6),appearance = list(rhs = c("pep=YES")))

#Sorting stuff out
#```{r}
myRules<-sort(myRules, by="confidence", decreasing=TRUE)
#```
summary(myRules)

# Show the top 5 rules, but only 2 digits
#```{r}
options(digits=2)
inspect(myRules[1:10])

summary(myRules)
#```


# Plot visulizations
#```{r}
arulesViz::ruleExplorer(myRules)
arulesViz::plot(myRules,method="graph",engine = "interactive",shading=NA)
#```


myRules = apriori(MyStoryData.P2, parameter = list(supp = 0.1, conf = 0.5, maxlen =6))

#Sorting stuff out
#```{r}
myRules<-sort(myRules, by="confidence", decreasing=TRUE)
#```


# Show the top 5 rules, but only 2 digits
#```{r}
options(digits=2)
inspect(myRules[1:10])

summary(myRules)
#```


# Plot visulizations
#```{r}
arulesViz::ruleExplorer(myRules)
arulesViz::plot(myRules,method="graph",engine = "interactive",shading=NA)
#```
