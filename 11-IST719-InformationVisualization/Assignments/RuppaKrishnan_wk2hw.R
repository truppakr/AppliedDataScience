
###########################################
#
# Author : Ram Krishnan
# Purpose: Week 2 Homework
#
###########################################


# Part 1 : Visualization This : Bar Plot, Stacked Bars, Scatterplot, Time Series, Step Chart

hotdogs <- read.csv("http://datasets.flowingdata.com/hot-dog-contest-winners.csv"
                    , header = TRUE
                    , stringsAsFactors = FALSE)

colnames(hotdogs)

head(hotdogs)

barplot(hotdogs$Dogs.eaten,names.arg = hotdogs$Year,col="red"
        , xlab = "Year"
        , ylab = "Hot dogs and buns (HDB) eaten")

fill_colors<- rep("#cccccc",nrow(hotdogs))

fill_colors[hotdogs$New.record=="1"] <- "#821122"

barplot(hotdogs$Dogs.eaten,names.arg = hotdogs$Year,col=fill_colors
        , border = NA
        , space = 0.3
        , xlab = "Year"
        , ylab = "Hot dogs and buns (HDB) eaten"
        , main = "Nathan's Hot Dog Eating Contest Results, 1980-2010")


hot_dog_places <- read.csv("http://datasets.flowingdata.com/hot-dog-places.csv"
                           , header = TRUE
                           , stringsAsFactors = FALSE)


names(hot_dog_places) <- as.character(c(2000:2010))

hot_dog_matrix <- as.matrix(hot_dog_places)

barplot(hot_dog_matrix,border = NA, space = 0.25, ylim = c(0,200)
        , xlab = "Year"
        , ylab = "Hot dogs and buns (HDBs) eaten"
        , main = "Hot Dog Eating Contest Results, 1980-2010")



subscribers <- read.csv("http://datasets.flowingdata.com/flowingdata_subscribers.csv"
                        , header = TRUE
                        , stringsAsFactors = FALSE)


subscribers[1:5,]

plot(subscribers$Subscribers)

plot(subscribers$Subscribers,type = "p", ylim = c(0,30000))

plot(subscribers$Subscribers,type = "h", ylim = c(0,30000), xlab = "Day"
     , ylab = "Subscribers")
points(subscribers$Subscribers,pch = 19, col="black")




population <- read.csv("http://datasets.flowingdata.com/world-population.csv"
                       , header = TRUE
                       , stringsAsFactors = FALSE)


population[1:5,]

plot(population$Year, population$Population
     , type = "l"
     , ylim = c(0,7000000000)
     , bty = "n"
     , xlab = "Year"
     , ylab = "Population")



postage <- read.csv("http://datasets.flowingdata.com/us-postage.csv"
                       , header = TRUE
                       , stringsAsFactors = FALSE)
postage[1:5,]

plot(postage$Year, postage$Price, type = "s"
     , xlab = "Year"
     , ylab = "Postage Rate (Dollars)"
     , main = "US Postage Rates for Letters, First Ounce, 1991-2010")

# Part 2 : Simple Distribution

file_nm <- file.choose()
art <- read.csv(file = file_nm, header = TRUE
                , stringsAsFactors = FALSE)

art[1:5,]

hist(art$total.sale 
     , xlab = "Total Sales"
     , main = "Distribution of total.sales"
     , col ="orange"
     , border = NA)


boxplot(art$total.sale
        , xlab = "Total Sales"
        , main = "Distribution of total.sales"
        , col = "orange")

art.drawing <- art[art$paper=="drawing",]
art.watercolor <- art[art$paper=="watercolor",]

d.drawing <- density(art.drawing$total.sale)
d.watercolor <- density(art.watercolor$total.sale)

plot(d.drawing
     , xlab = "Total Sales"
     , main = "Distribution of total sales for the drawing paper"
     , col ="orange"
     , border = NA)

polygon(d.drawing
        , xlab = "Total Sales"
        , main = "Distribution of total sales for the drawing paper"
        , col ="orange"
        , border = NA)


plot(d.watercolor
     , xlab = "Total Sales"
     , main = "Distribution of total sales for the watercolor paper"
     , col ="orange"
     , border = NA)

polygon(d.watercolor
     , xlab = "Total Sales"
     , main = "Distribution of total sales for the watercolor paper"
     , col ="blue"
     , border = NA)

library(vioplot)
vioplot(art$total.sale)
vioplot(art.drawing$total.sale)
vioplot(art.$total.sale)

my.par <- par()


par(mfrow=c(2,2))



hist(art$total.sale 
     , xlab = "Total Sales"
     , main = "Distribution of total.sales"
     #, col = rgb(143,253,200, maxColorValue = 255)
     , col = "orange"
     , border = NA)


boxplot(art$total.sale
        , xlab = "Total Sales"
        , main = "Distribution of total.sales"
        , col = rgb(209,150,253, maxColorValue = 255))


plot(d.drawing
     , xlab = "Total Sales"
     , main = "Distribution of total sales for the drawing paper"
     #, col = rgb(255,251,144, maxColorValue = 255)
     , col = "blue"
     , border = NA)

polygon(d.drawing
        , xlab = "Total Sales"
        , main = "Distribution of total sales for the drawing paper"
        #, col = rgb(255,251,144, maxColorValue = 255)
        , col = "blue"
        , border = NA)


plot(d.watercolor
     , xlab = "Total Sales"
     , main = "Distribution of total sales for the watercolor paper"
     , col = rgb(255,172,144, maxColorValue = 255)
     , border = NA)

polygon(d.watercolor
        , xlab = "Total Sales"
        , main = "Distribution of total sales for the watercolor paper"
        , col = rgb(255,172,144, maxColorValue = 255)
        , border = NA)


View(art)
par(my.par)

# Part 3 : Grouping and Multidimensional Plots

unit.price.sold <- tapply(art$units.sold,list(art$unit.price),sum)


layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))

# store the current value of adj
adj.old <- par()$adj    # default is 0.5


plot(unit.price.sold, pch = 19 , cex=1
     , xaxt = "n"
     , col = "blue"
     , bty ="n"
     , xlab = "Unit Price"
     , ylab = "Units Sold"
     , main = "Unit Price Vs Units Sold"
     , adj = 0
     )
axis(1, at=1:8, labels=sort(unique(art$unit.price)))

# set adj to right-aligned and plot the subtitle
par(adj = 0.5)
title(sub = "Unit price doesn't seem to correlate with units sold")

# reset adj to previous value
par(adj = adj.old)

abline(lm(unit.price.sold ~ sort(unique(art$unit.price))),col = "red", lwd = 2, lty =2)


drawing.watercolor.sold <- tapply(art$units.sold,list( art$paper,art$year),sum)

my.color= c(rgb(206,89,115, maxColorValue = 255),rgb(28,186,188,maxColorValue = 255))

barplot(drawing.watercolor.sold
     , beside = TRUE
     , col = my.color
     , bty ="n"
     , xlab = "Year"
     , ylab = "Units sold"
     , ylim = c(0,5000)
     , main = "Units sold for drawing and watercolor"
     , adj = 0
     , border = NA
     , las = 2
     #, space = 0.3
     #,legend.text = c("drawing","Watercolor")
)

# Add a legend
legend(1,5000, inset=.02, c("drawing","watercolor")
       , bty = "n"
       , border = NA
       , fill=my.color
       , horiz=TRUE
       , cex=1.2)



title(sub = "Units sold for watercolor is more than drawing across all years")



drawing.watercolor.income <- tapply((art$units.sold*art$unit.price),list( art$paper,art$year),sum)


barplot(drawing.watercolor.income
        , beside = TRUE
        , col = my.color
        , bty ="n"
        , xlab = "Year"
        , ylab = "Income"
        , ylim = c(0,40000)
        , main = "Income from drawing and watercolor paper"
        , adj = 0
        , border = NA
        
        #,legend.text = c("drawing","Watercolor")
)

# Add a legend
legend(1,40000, inset=.02, c("drawing","watercolor")
       , bty = "n"
       , border = NA
       , fill=my.color
       , horiz=TRUE
       , cex=1.2)



title(sub = "Income from watercolor is more than drawing paper across all years")



# Part 4 : Additional charts for week4 HW



unemployment <- read.csv("http://datasets.flowingdata.com/unemployment-rate-1948-2010.csv"
                         , header = TRUE
                         , stringsAsFactors = FALSE )
View(unemployment)

plot(1:length(unemployment$Value), unemployment$Value)
scatter.smooth(x=1:length(unemployment$Value),y=unemployment$Value
               ,ylim = c(0,11), degree = 2, col="#CCCCCC",span = 0.5)
