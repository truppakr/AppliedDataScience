
###########################################
#
# Author : Ram Krishnan
# Purpose: Week 4 Homework
#
###########################################


crime <- read.csv("http://datasets.flowingdata.com/crimeRatesByState2005.csv"
                  , header = TRUE
                  , stringsAsFactors = FALSE)
View(crime)

crime2 <- crime[crime$state != "District of Columbia", ]
crime2 <- crime2[crime2$state != "United States", ]

View(crime2)

pairs(crime2[,2:9], panel = panel.smooth, main = "Crime Rates per 100,000 Population")

# Open a pdf file
pdf("crimerates_pairs.pdf", width = 10, height = 10) 
# 2. Create a plot
pairs(crime2[,2:9], panel = panel.smooth, main = "Crime Rates per 100,000 Population")
# Close the pdf file
dev.off() 

symbols(crime2$murder,crime2$burglary,circles = crime2$population)

radius <- sqrt(crime2$population/pi)
symbols(crime2$murder,crime2$burglary,circles = radius
        , inches = 0.35
        , fg = "white"
        , bg = "red"
        , xlab = "Murder Rate"
        , ylab = "Burglary Rate"
        , main = "Murders Versus Burglaries in the Unites States per 100,000 population"
        )
text(crime2$murder, crime2$burglary, crime2$state, cex = 0.5)



birth <- read.csv("http://datasets.flowingdata.com/birth-rate.csv"
                  , header = TRUE
                  , stringsAsFactors = FALSE)
stem(birth$X2008)

hist(birth$X2008, breaks = 5)

hist(birth$X2008, breaks = 10
     , xlab = "Live berths per 1,000 Population"
     , main = "Global Distribution of birth rates"
     , col =rgb(160,71,125,maxColorValue=255)
     )


birth2008 <- birth$X2008[!is.na(birth$X2008)]

d2008 <- density(birth2008)

density.default(x=birth2008)

d2008frame <- data.frame(d2008$x,d2008$y)
write.table(d2008frame,"birthdensity.txt", sep = "\t")

write.table(d2008frame,"birthdensity.txt", sep = ",", row.names = FALSE)

plot(d2008, type = "n"
     , main = "Global Distribution Of Birth Rates in 2008"
     , xlab = "Live births per 1,000 population")
polygon(d2008, col="#821122", border = "#cccccc")


library(lattice)
histogram(birth$X2008,breaks=10)
lines(d2008)


birth_yearly <- read.csv("http://datasets.flowingdata.com/birth-rate-yearly.csv"
                  , header = TRUE
                  , stringsAsFactors = FALSE)

histogram(~rate | year, data= birth_yearly, layout=c(10,5))

us.health <- read.csv(file = "C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST719\\Week4\\county_health_rankings\\county_health_rankings\\us-county-health-rankings-2020.csv", header = T
                      , stringsAsFactors = F)



us.health.states    <- us.health[us.health$county=='',]
us.health.counties  <- us.health[us.health$county!='',]

histogram(~population | state, data= us.health.counties, layout=c(10,6)
          , main ="Distribution of Population in US States")

