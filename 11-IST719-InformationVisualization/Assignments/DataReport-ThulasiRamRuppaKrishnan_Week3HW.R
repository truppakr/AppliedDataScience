###########################################################################
#
# Author : Ram Krishnan
# Purpose: Week 3 HW
#
###########################################################################
library(dplyr)
library(corrplot)
library(scales)
library(vioplot)

file_nm <- file.choose()
my.par <- par()

#####################   Function Definitions   ############################

# function(x){(x-min(x))/(max(x)-min(x))}

std <- function(observed) {
  
  rescale(observed, to = c(-1,1))
}

# Test if the function is working fine
# std(1:100)



# mat : is a matrix of data
# ... : further arguments to pass to the native R cor.test function
cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

# matrix of the p-value of the correlation




###########################################################################


# Part 1 : Data Mining and Correlations

us.health <- read.csv(file = file_nm, header = T
                      , stringsAsFactors = F)

head(us.health)
colnames(us.health)

us.health.states    <- us.health[us.health$county=='',]
us.health.counties  <- us.health[us.health$county!='',]

View(us.health.states)
View(us.health.counties)

str(us.health.states)

state_cols <- as.data.frame(ifelse(sapply(us.health.states, function(x)any(is.na(x))) == TRUE, "Y","N")) 
state_cols <- `colnames<-`(state_cols,c('NotAvailable'))
state_cols$metric <- row.names(state_cols)
row.names(state_cols) <- NULL

us.health.states <- us.health.states[,state_cols[state_cols$NotAvailable=='N',]$metric]
# View(us.health.states)


# Drop column names of the dataframe that are non numeric
us.health.states <- select_if(us.health.states, is.numeric)
us.health.states <- select(us.health.states,-starts_with("x95"))
us.health.states.s <- us.health.states[,4:dim(us.health.states)[2]] %>% mutate_all(std)
# View(us.health.states.s)
corr_cols <- as.data.frame(colnames(us.health.states.s)) 
# us.health.states.s <- `colnames<-`(us.health.states.s,as.character(c(1:dim(us.health.states.s)[2])))

us.health.states.m <- as.matrix(us.health.states.s)
# str(us.health.states.m)


# Open a pdf file
pdf("heat_map.pdf", width = 20, height = 40) 
# 2. Create a plot
corrplot(us.health.states.m,method = 'color')
# Close the pdf file
dev.off() 



us.health.states.c <- cor(us.health.states.s)
# str(us.health.states.c)

p.mat <- cor.mtest(us.health.states.s)
head(p.mat[, 1:5])



# Open a pdf file
pdf("correlation_matrix.pdf", width = 30, height = 30) 
# 2. Create a plot
corrplot(us.health.states.c,method = 'color',type = 'upper'
         , order = 'hclust', col=colorRampPalette(c('red', 'white', 'blue'))(20)
         , p.mat = p.mat, sig.level = 0.01, insig = 'blank')
# Close the pdf file
dev.off() 



# Part 2 : Simple Distribution

par(my.par)

hist(us.health.states$population
     , xlab = "Population"
     , main = "Distribution of State Population"
     , col ="orange"
     , border = NA)

hist(us.health.counties$population
     , xlab = "Population"
     , main = "Distribution of County Population"
     , col ="orange"
     , border = NA)




boxplot(us.health.states$population
        , xlab = "State Population"
        , main = "Distribution of State Population"
        , col = "orange")

boxplot(us.health.counties$population
        , xlab = "County Population"
        , main = "Distribution of County Population"
        , col = "orange")

ds <- density(us.health.states$population)
dc <- density(us.health.counties$population)

plot(ds
     , xlab = "State Population"
     , main = "Distribution of state population"
     , col ="orange"
     , border = NA)

polygon(ds
        , xlab = "State Population"
        , main = "Distribution of state population"
        , col ="orange"
        , border = NA)


plot(dc
     , xlab = "County Population"
     , main = "Distribution of county population"
     , col ="orange"
     , border = NA)

polygon(dc
        , xlab = "County Population"
        , main = "Distribution of county population"
        , col ="orange"
        , border = NA)


vioplot(us.health.states$population)
vioplot(us.health.counties$population)



dpu <- density(us.health.counties$average_number_of_physically_unhealthy_days)
dmu <- density(us.health.counties$average_number_of_mentally_unhealthy_days)


par(mfrow=c(2,2))
par(my.par)

# store the current value of adj
adj.old <- par()$adj    # default is 0.5


hist(us.health.counties$median_household_income 
     , xlab = "Median household income"
     , main = "Distribution of Median household income"
     , col = "orange"
     , adj = 0
     , border = NA)

# set adj to right-aligned and plot the subtitle
par(adj = 0)

title(sub = "Context : This plot shows the distribution of Median household income for counties in USA ", line = -19, cex.sub = 1)
title(sub = "Data Source : https://www.kaggle.com/roche-data-science-coalition/uncover ", line = 4, cex.sub =0.8, font.sub = 3)


par(adj = adj.old)
boxplot(us.health.counties$life_expectancy
        , xlab = "Life expectancy"
        , main = "Distribution of Life expectancy"
        , col = "blue")


# set adj to right-aligned and plot the subtitle
par(adj = 0)

title(sub = "Context : This plot shows the distribution of life expectancy in years for counties in USA ", line = -19, cex.sub = 1)
title(sub = "Data Source : https://www.kaggle.com/roche-data-science-coalition/uncover ", line = 4, cex.sub =0.8, font.sub = 3)



plot(dpu
     , xlab = "Average number of physically unhealthy days"
     , main = "Distribution of Average physically unhealthy days"
     , col =rgb(209,150,253, maxColorValue = 255)
     , bty = "n"
     , adj = 0
     , border = NA)

polygon(dpu
        , xlab = "Average number of physically unhealthy days"
        , main = "Distribution of Average physically unhealthy days"
        , col =rgb(209,150,253, maxColorValue = 255)
        , bty = "n"
        , adj = 0
        , border = NA)


title(sub = "Context : This plot shows the distribution of Average number of physically unhealthy days for counties in USA "
      , line = -19, cex.sub = 1)
title(sub = "Data Source : https://www.kaggle.com/roche-data-science-coalition/uncover "
      , line = 4, cex.sub =0.8, font.sub = 3)



plot(dmu
     , xlab = "Average number of mentally unhealthy days"
     , main = "Distribution of Average mentally unhealthy days"
     , col = rgb(255,172,144, maxColorValue = 255)
     , bty = "n"
     , adj = 0
     , border = NA)

polygon(dmu
        , xlab = "Average number of mentally unhealthy days"
        , main = "Distribution of Average mentally unhealthy days"
        , col = rgb(255,172,144, maxColorValue = 255)
        , bty = "n"
        , adj = 0
        , border = NA)


title(sub = "Context : This plot shows the distribution of Average number of mentally unhealthy days for counties in USA "
      , line = -19, cex.sub = 1)
title(sub = "Data Source : https://www.kaggle.com/roche-data-science-coalition/uncover "
      , line = 4, cex.sub =0.8, font.sub = 3)



# reset adj to previous value
par(adj = adj.old)

View(art)
par(my.par)


# Part 3 : Multidimension plots


# percent_fair_or_poor_health
# percent_limited_access_to_healthy_foods
# percent_low_birthweight
# infant_mortality_rate
# child_mortality_rate

us.health.states.fb <- select(us.health[us.health$county=='',]
                              ,c("state"
                                 ,"percent_fair_or_poor_health"
                                 ,"percent_limited_access_to_healthy_foods"
                                 ,"percent_low_birthweight"
                                 ,"infant_mortality_rate"
                                 ,"child_mortality_rate"))
#tapply(us.health.states.fb,list(us.health.states.fb$state),sum)

View(us.health.states.fb)

my.color4= c(rgb(78,205,196,maxColorValue = 255)
             ,rgb(199,244,100,maxColorValue = 255)
             ,rgb(255,107,107,maxColorValue = 255)
             #, rgb(196,77,88,maxColorValue = 255)
             #, rgb(85,98,112, maxColorValue = 255)
             )

barplot( cbind(us.health.states.fb$percent_limited_access_to_healthy_foods
               , us.health.states.fb$percent_low_birthweight
               , us.health.states.fb$infant_mortality_rate
               #, us.health.states.fb$child_mortality_rate
               #, us.health.states.fb$percent_fair_or_poor_health
               ) ~ us.health.states.fb$state
        , beside = TRUE
        , srt       = 60
        , col = my.color4
        , bty ="n"
        , xlab = "\n\nState"
        , ylab = "Percentage"
        , ylim = c(0,15)
        , main = "Children health metrics in percentage"
        , adj = 0
        , border = NA
        , horiz = F
        , las = 2
        , cex.axis = 0.8
        , cex.names = 0.8
        
        # , legend.text = TRUE
        # , args.legend = list(x = "topright", bty = "n", inset=c(-0.15, 0))
)



# Add a legend
legend(150,15, inset=.02, c("Limited access to healthy foods"
                            ,"Low birthweight"
                            ,"Infant mortality rate"
                            #,"child_mortality_rate"
                            #,"percent_fair_or_poor_health"
                            )
       , bty = "n"
       , border = NA
       , fill=my.color4
       , horiz=F
       , cex=1)


title(sub = " Context : This plot compares the percentage of population with poor health , percentage of \
 of limited access to healthy foods with birth weight and child/infant mortality rate  \
 acrosss all states in USA "
      , line = -24, cex.sub = 1, adj = 0)
title(sub = "\nData Source : https://www.kaggle.com/roche-data-science-coalition/uncover "
      , line = 4, cex.sub =0.8, font.sub = 3, adj = 0)





# percent_excessive_drinking
# percent_driving_deaths_with_alcohol_involvement
# percent_drive_alone_to_work
# motor_vehicle_mortality_rate

us.health.states.ad <- select(us.health[us.health$county=='',]
                              ,c("state"
                                 ,"percent_excessive_drinking"
                                 ,"percent_driving_deaths_with_alcohol_involvement"
                                 ,"percent_drive_alone_to_work"
                                  ))


barplot( cbind(us.health.states.ad$percent_excessive_drinking
               , us.health.states.ad$percent_driving_deaths_with_alcohol_involvement
               , us.health.states.ad$percent_drive_alone_to_work

) ~ us.health.states.ad$state
, beside = TRUE
, srt       = 60
, col = my.color4
, bty ="n"
, xlab = "\n\nState"
, ylab = "Percentage"
, ylim = c(0,100)
, main = "Excessive Alcohol usage and Driving conditions"
, adj = 0
, border = NA
, horiz = F
, las = 2
, cex.axis = 0.8
, cex.names = 0.8

# , legend.text = TRUE
# , args.legend = list(x = "topright", bty = "n", inset=c(-0.15, 0))
)



# Add a legend
legend(150,100, inset=.02, c("Excessive drinking"
                            ,"Driving death with alochol involvement"
                            ,"Drive alone to work"

)
, bty = "n"
, border = NA
, fill=my.color4
, horiz=F
, cex=1)


title(sub = " Context : This plot compares the percentage of excessive alcohol drinkers  \
      and the percentage of workers driving alone with driving deaths  \
      acrosss all states in USA "
      , line = -24, cex.sub = 1, adj = 0)
title(sub = "\nData Source : https://www.kaggle.com/roche-data-science-coalition/uncover "
      , line = 4, cex.sub =0.8, font.sub = 3, adj = 0)




# percent_adults_with_obesity
# percent_physically_inactive
# percent_adults_with_diabetes

head(us.health.counties)
histogram(~population | state, data= us.health.counties, layout=c(10,6)
          , main ="Distribution of Population in US States")


