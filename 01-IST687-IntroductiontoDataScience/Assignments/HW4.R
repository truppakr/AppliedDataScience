printVecInfo <- function(vc)
{
  library(moments)
  print(paste("Mean :" ,mean(vc)))
  print(paste("Median :" ,median(vc)))
  print(paste("Min :" ,min(vc)," Max :" ,max(vc)))
  print(paste("sd :" ,sd(vc)))
  print(paste("quantile (0.05-0.95) :" ,paste(quantile(vc,c(0.05,0.95)),collapse = " - ")))
  print(paste("Skewness :" ,skewness(vc)))
}

jar <-as.character(replicate(50,sample(c("red","blue"),2,replace = FALSE),simplify = "FALSE"))

s1 <- sample(jar,10,replace = TRUE)
length(which(s1=="red"))
(length(which(s1=="red"))/10)*100

s2 <- as.numeric(sample(jar,10,replace = TRUE)=="red")
sum(s2)
sum(s2)/length(s2)

samplemean_20_10<-replicate(20,mean(as.numeric(sample(jar,10,replace = TRUE)=="red")),simplify = "FALSE")

printVecInfo(samplemean_20_10)

hist(samplemean_20_10)


samplemean_20_100<-replicate(20,mean(as.numeric(sample(jar,100,replace = TRUE)=="red")),simplify = "FALSE")

printVecInfo(samplemean_20_100)

hist(samplemean_20_100)


samplemean_100_100<-replicate(100,mean(as.numeric(sample(jar,100,replace = TRUE)=="red")),simplify = "FALSE")

printVecInfo(samplemean_100_100)

hist(samplemean_100_100)


myAq<-airquality

myCleanAq<-myAq[which((myAq$Ozone=="NA" | myAq$Solar.R=="NA")=="FALSE"),]

printVecInfo(myCleanAq$Ozone)
hist(myCleanAq$Ozone)
printVecInfo(myCleanAq$Wind)
hist(myCleanAq$Wind)
printVecInfo(myCleanAq$Temp)
hist(myCleanAq$Temp)





