
##################################################
#
# Author : Ram Krishnan
# Purpose: Week 5 Lab:: Working with Twitter Data
# Uses: ClimateTweets_UseForLecture_25k.csv
#
##################################################


my.dir <- "C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST719\\Week5\\"

tweets <- read.csv(file = paste0(my.dir,"ClimateTweets_UseForLecture_25k.xls")
                   , header = TRUE
                   , quote = "\""
                   , stringsAsFactors = FALSE)

View(tweets)

my.media <- tweets$media

table(my.media)

my.media[my.media==""] <- "text only"

my.media <- gsub("\\|photo","",my.media)


round(table(my.media)/sum(table(my.media)),4)

pie(100 * round(table(my.media)/sum(table(my.media)),4))


tweets$created_at[1:3]

# "Wed Jul 06 03:35:37 +0000 2016"

converstion.string <- "%a %b %d %H:%M:%S +0000 %Y"


tmp <- strptime(tweets$created_at[1:3],converstion.string)

class(tmp)


tmp <- strptime(tweets$created_at,converstion.string)

any(is.na(tmp))

tweets$date <- strptime(tweets$created_at,converstion.string)

tmp <- "10AM and 27 minutes, on June 22, 1999"

strptime(tmp,"%H%p and %M minutes, on %B %d, %Y")

min(tweets$date)
max(tweets$date)
range(tweets$date)
summary(tweets$date)

my.df <- data.frame()
my.df <- data.frame(cbind( sample(c("T","H","Q"), 739, replace = T), sample(1:100, 739, replace = T),as.numeric(sample(1:1000, 739, replace = T)) ))
# my.df$date.time <- tweets$date[1:739]
colnames(my.df) <- c("mode","x","y"
                     #,"date.time"
                     )
my.df$x <- as.numeric(my.df$x)
my.df$y <- as.numeric(my.df$y)
head(my.df)
dim(my.df)
str(my.df)
ggplot(my.df, aes(x=x, y=y, color=mode, alpha=.5)) + geom_point() +
  scale_color_manual(values=c(rgb(144,238,144,maxColorValue = 255),
                              rgb(139,0,0,maxColorValue = 255),
                              rgb(128,0,128,maxColorValue = 255)))



plot(x=my.df$x,y=my.df$y, col)

difftime(min(tweets$date),max(tweets$date))

difftime(min(tweets$date),max(tweets$date), units = "mins")

difftime(min(tweets$date),max(tweets$date), units = "weeks")

library(lubridate)

wday(tweets$date[1:3],label = TRUE, abbr= TRUE)

barplot(table(wday(tweets$date,label = TRUE, abbr= TRUE)))

tmp <- tweets$user_utc_offset

tweets$date[7:10] + tmp[7:10]

known.times <- tweets$date + tweets$user_utc_offset

index <- which(is.na(known.times))

known.times <- known.times[-index]

barplot(table(hour(known.times)))

"2018.08.30-16.24.49"

strptime("2018.08.30-16.24.49","%Y.%m.%d-%H.%M.%S")


strptime("2014, Aug, Fri the 16 at 18:40","%Y, %b, %a the %d at %H:%M")

sample <- c("2014, Aug, Fri the 16 at 18:40","2014, Jun, Sat the 24 at 11:51","2014, Jun, Sun the 25 at 7:22")
strptime(sample,"%Y, %b, %a the %d at %H:%M")

start.date <- as.POSIXct("2016-06-24 23:59:59")
end.date <- as.POSIXct("2016-06-26 00:00:00")

index <- which((tweets$date>start.date) & (tweets$date<end.date))

tweets.25th <- tweets$date[index]

format.Date(tweets.25th,"%Y%m%d%H%M")

tmp.date <- as.POSIXct(strptime(format.Date(tweets.25th,"%Y%m%d%H%M")
                               ,"%Y%m%d%H%M"))

plot(table(tmp.date))

length(table(tmp.date))
24*60


tmp.tab <- table(tmp.date)


plot(as.POSIXct(names(tmp.tab)), as.numeric(tmp.tab), type ="h")


x <- seq.POSIXt(from = start.date +1, to = end.date - 1, by ="min")
x

y <- rep(0, length(x))
y[match(names(tmp.tab), as.character(x))] <- as.numeric(tmp.tab)

plot(x,y, type = "p", pch = ".", cex =.4)
plot(x,y, type = "l")


seq.POSIXt()


seq.POSIXt(from = as.POSIXct("2020-01-01"), to = as.POSIXct("2020-03-31"), by ="day")

plot(x,y, type = "p", pch = 16 , cex =.4)

####################################################################################################3
# hashtag word cloud
####################################################################################################3


tweets$text[5:10]

library(stringr)
tags <- str_extract_all(tweets$text, "#\\S+", simplify = FALSE)


tags <- tags[lengths(tags)>0]

tags <- unlist(tags)

tags <- tolower(tags)
tags <- gsub("#|[[:punct:]]","",tags)

tag.tab <- sort(table(tags), decreasing = TRUE)

tag.tab[1:10]

zap <- which(tag.tab<3)

tag.tab <- tag.tab[-zap]

plot(as.numeric(tag.tab))

df <- data.frame(words = names(tag.tab), count=as.numeric(tag.tab)
                 , stringsAsFactors = FALSE)


par(mfrow = c(3,3))

plot(df$count, main="raw")

y <- df$count/max(df$count)

plot(y, main = "0 - 1")
plot(df$count^(1/2), main = "^(1/2)")
plot(df$count^(1/5), main = "^(1/5)")
plot(log10(df$count), main = "log10")
plot(log(df$count), main = "log")


library(wordcloud)

mpal <- colorRampPalette(c("gold","red","orange"))
mpal <- colorRampPalette(c("red","orange3","gold"))

gc()

index <- which(df$count>8)

par(mar=c(0,0,0,0), bg="black")

my.counts <- (df$count[index])^(1/2)
wordcloud(df$words[index],my.counts, scale= c(5,.5), min.freq = 1
          , max.words = Inf, random.order = FALSE
          , random.color = FALSE,  ordered.colors = TRUE
          , rot.per = 0, colors = mpal(length(df$words[index])))



#########################################################################################
# Alluvial plots & treemap plots
########################################################################################


my.dir

sales <- read.csv(file = paste0(my.dir,"sales.xls")
                  , header = TRUE
                  , stringsAsFactors = FALSE)

install.packages("alluvial")
library(alluvial)

dat <- as.data.frame(Titanic, stringsAsFactors = FALSE)
alluvial(dat[,1:4], freq = dat$Freq)


alluv.df <- aggregate(sales$units.sold
                      , list(sales$rep.region, sales$type)
                      , sum)

colnames(alluv.df) <-  c("reg","type","units.sold")

alluvial(alluv.df[,1:2],freq = alluv.df$units.sold)
my.cols <- rep("gold",nrow(alluv.df))
my.cols[alluv.df$type=="red"] <- "red"

alluvial(alluv.df[,1:2],freq = alluv.df$units.sold, col = my.cols)


alluvial(alluv.df[,1:2],freq = alluv.df$units.sold, 
         col = ifelse(alluv.df$type=="red","red","gold"))


options(stringsAsFactors = FALSE)



alluv.df <- aggregate(sales$units.sold
                      , list(sales$rep.region
                             , sales$type
                             , sales$wine)
                      , sum)
colnames(alluv.df) <- c("reg","type","wine","untis.sold")

alluvial(alluv.df[,1:3], freq = alluv.df$untis.sold
         , col = ifelse(alluv.df$type=="red","red","gold")
         , alpha= 0.3
         , gap.width= 0.1)


library(RColorBrewer)

install.packages("treemap")
library(treemap)

treemap(sales, index = c("rep.region")
        , vSize = "income"
        , fontsize.labels = 18
        , palette = "Greens")

treemap(sales, index = c("rep.region")
        , vSize = "income"
        , vColor = "units.sold"
        , type = "dens"
        , fontsize.labels = 18
        , palette = "Greens")


treemap(sales, index = c("rep.region")
        , vSize = "income"
        , vColor = "units.sold"
        , type = "value"
        , fontsize.labels = 18
        , palette = "OrRd")



treemap(sales, index = c("rep.region", "sales.rep", "type")
        , vSize = "income"
        , vColor = "units.sold"
        , type = "value"
        , fontsize.labels = 18
        , palette = brewer.pal(8,"Set1"))



####################################################################################################
# River Plot and figure out new plots
####################################################################################################



install.packages("riverplot")
library("riverplot")


river <- riverplot.example()
par(mfrow = c(2,1), mar = c(1,1,1,1))

plot(river, srt=90, lty=1)
class(river)

x <- river
x$edges

x$edges$Value[2] <- 45
x$edges$Value[1] <- 15

x$nodes$x[5] <- 5

plot(x)


df <- aggregate(sales$income,
                list(type = sales$type, wine= sales$wine)
               , sum)


df <- df[order(df$type,df$x),]

node.name <- c("wine",unique(df$type), df$wine)
node.position <- c(1, 2,2, 3,3,3,3,3,3,3)
node.color <- rep("gray", length(node.name))
node.color <- c("deepskyblue", "red", "yellow"
                , "brown4", "firebrick3", "deeppink4"
                , "khaki1", "lightgoldenrod1", "gold", "goldenrod1")



node <- data.frame(ID = node.name
                   , x = node.position
                   , col = node.color
                   , stringsAsFactors = FALSE)



parent.nodes <- c("wine","wine",df$type)

child.nodes <- c("red","white",df$wine)


value <- c(sum(df$x[df$type=="red"]), sum(df$x[df$type=="white"]),df$x)

edges <- data.frame(N1 = parent.nodes, N2 = child.nodes, Value =value)

r <- makeRiver(node, edges, node_labels = node.name)

par(mar =c(0,0,0,0))
plot(r, srt=90, lty=1, cex=0.5)


####################################################################################################
# R plots and word
####################################################################################################


dat <- tapply(sales$units.sold, list(sales$type,sales$rep.region), sum)

barplot( dat, beside = TRUE, col =c("brown","gold")
         , main = "Units sold by region by type")






























































































