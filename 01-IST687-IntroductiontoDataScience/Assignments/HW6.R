
install.packages("stringr")
install.packages("scales")
install.packages('dplyr')
install.packages('ggplot2')
install.packages('lubridate')

library(ggplot2)
library(reshape2)
library(stringr)
library(scales)
library(plyr)
library(lubridate)

my_aq<- airquality
my_aq<-na.omit(my_aq)
my_aq$Month<-as.factor(my_aq$Month)
my_aq$Day<-as.factor(my_aq$Day)
my_aq<-cbind.data.frame(my_aq,"date"=as.Date(gsub(" ","",paste(str_pad(my_aq$Month,2,side="left",pad = "0"),"-",str_pad(my_aq$Day,2,side="left",pad = "0"),"-1973")),"%m-%d-%Y"))
my_aq.m <- melt(my_aq,id.vars = "date", measure.vars = c("Ozone", "Solar.R","Wind","Temp"))
my_aq.m <- ddply(my_aq.m, .(variable), transform, rescale = rescale(value))

ggplot(data = melt(my_aq,measure.vars = c("Ozone", "Solar.R","Wind","Temp")), mapping = aes(x = value)) + geom_histogram(bins = 20) + facet_wrap(~variable, scales = 'free_x')
ggplot(my_aq,aes(x=date , y=Ozone, group=Month ,fill=format.Date(date,"%B"))) + geom_boxplot() + geom_dotplot(binaxis='y', stackdir='center', dotsize=0.5, binwidth = 7,fill="Black") +scale_fill_brewer(palette="Set2") +scale_x_date(labels = date_format("%B")) +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
   labs(x = "Month",title = "Ozone Boxplot over Month")  + guides(fill=guide_legend(title=NULL))
ggplot(my_aq,aes(x=date , y=Wind, group=Month ,fill=format.Date(date,"%B"))) + geom_boxplot() + geom_dotplot(binaxis='y', stackdir='center', dotsize=0.05, binwidth = 7,fill="Black") +scale_fill_brewer(palette="Set2") +scale_x_date(labels = date_format("%B")) +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Wind Boxplot over Month")  + guides(fill=guide_legend(title=NULL))

ggplot(my_aq,aes(x=date , y=Wind ,group=1)) +   geom_line(color="red",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Wind speed over days")  
ggplot(my_aq,aes(x=date , y=Wind ,group=1)) +   geom_smooth(method="lm",color="red",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Wind speed over days using linear method") 
ggplot(my_aq,aes(x=date , y=Wind ,group=1)) +   geom_smooth(method="loess",color="red",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Wind speed over days using loess method") 

ggplot(my_aq,aes(x=date , y=Ozone ,group=1)) +   geom_line(color="blue",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Ozone over days") 
ggplot(my_aq,aes(x=date , y=Ozone ,group=1)) +   geom_smooth(method="lm",color="blue",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Ozone over days using linear method") 
ggplot(my_aq,aes(x=date , y=Ozone ,group=1)) +   geom_smooth(method="loess",color="blue",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Ozone over days usng loess method") 

ggplot(my_aq,aes(x=date , y=Solar.R ,group=1)) +   geom_line(color="#FF9999",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Solar.R over days") 
ggplot(my_aq,aes(x=date , y=Solar.R ,group=1)) +   geom_smooth(method="lm",color="#FF9999",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Solar.R over days using linear method") 
ggplot(my_aq,aes(x=date , y=Solar.R ,group=1)) +   geom_smooth(method="loess",color="#FF9999",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Solar.R over days using loess method") 

ggplot(my_aq,aes(x=date , y=Temp ,group=1)) +   geom_line(color="purple",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Temparature over days") 
ggplot(my_aq,aes(x=date , y=Temp ,group=1)) +   geom_smooth(method="lm",color="purple",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Temperature over days using linear method") 
ggplot(my_aq,aes(x=date , y=Temp ,group=1)) +   geom_smooth(method="loess",color="purple",size=1) +scale_x_date(labels = date_format("%B")) + geom_point() +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Temperature over days using loess method") 

head(melt(my_aq,id.vars = "date", measure.vars = c("Ozone", "Solar.R","Wind","Temp")))
summary(my_aq$Ozone)
summary(my_aq$Wind)
summary(my_aq$Temp)
summary(my_aq$Solar.R)

ggplot(data = melt(my_aq,id.vars = "date", measure.vars = c("Ozone", "Solar.R","Wind","Temp")), aes(x=date,y=value,color=variable,group=1)) + geom_line(size=1) +scale_x_date(labels = date_format("%B"))+  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month", y="Value",title = "Air Quality over days") 


scaleFactor <- max(my_aq$Solar.R) / max(my_aq$Wind)

ggplot(my_aq, aes(x=date)) +
  geom_line(aes(y=Ozone),size=1, col="blue") + geom_point(aes(y=Ozone),size=1.25)+
  geom_line(aes(y=Solar.R), size=1, col="green")+ geom_point(aes(y=Solar.R),size=1.25) +
  geom_line(aes(y=Temp), size=1, col="black") + geom_point(aes(y=Temp),size=1.25)+
  geom_line(aes(y=Wind * scaleFactor), size=1, col="red")+ geom_point(aes(y=Wind * scaleFactor),size=1.25) +
  scale_y_continuous(name="Solar.R,Ozone,Temp", sec.axis=sec_axis(~./scaleFactor, name="Wind")) +
  theme(
    axis.title.y.left=element_text(color="blue"),
    axis.text.y.left=element_text(color="blue"),
    axis.title.y.right=element_text(color="red"),
    axis.text.y.right=element_text(color="red")
  ) +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Air Quality Over days on two scale Y axis") 

ggplot(my_aq, aes(x=date)) +
  geom_smooth(aes(y=Ozone),method="lm",size=1, col="blue") + 
  geom_smooth(aes(y=Solar.R),method="lm", size=1, col="green")+ 
  geom_smooth(aes(y=Temp),method="lm", size=1, col="black") +
  geom_smooth(aes(y=Wind * scaleFactor),method="lm", size=1, col="red")+ 
  scale_y_continuous(name="Solar.R,Ozone,Temp", sec.axis=sec_axis(~./scaleFactor, name="Wind")) +
  theme(
    axis.title.y.left=element_text(color="blue"),
    axis.text.y.left=element_text(color="blue"),
    axis.title.y.right=element_text(color="red"),
    axis.text.y.right=element_text(color="red")
  )+  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Air Quality Over days on two scale Y axis using linear method") 

ggplot(my_aq, aes(x=date)) +
  geom_smooth(aes(y=Ozone),method="loess",size=1, col="blue") + 
  geom_smooth(aes(y=Solar.R),method="loess", size=1, col="green")+ 
  geom_smooth(aes(y=Temp),method="loess", size=1, col="black") +
  geom_smooth(aes(y=Wind * scaleFactor),method="loess", size=1, col="red")+ 
  scale_y_continuous(name="Solar.R,Ozone,Temp", sec.axis=sec_axis(~./scaleFactor, name="Wind")) +
  theme(
    axis.title.y.left=element_text(color="blue"),
    axis.text.y.left=element_text(color="blue"),
    axis.title.y.right=element_text(color="red"),
    axis.text.y.right=element_text(color="red")
  )+  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Air Quality Over days on two scale Y axis using loess method") 



ggplot(my_aq, aes(x=date)) +
  geom_tile(aes(y=Ozone), col="blue") +
  geom_tile(aes(y=Solar.R), col="green")+ 
  geom_tile(aes(y=Temp), col="black") + 
  geom_tile(aes(y=Wind * scaleFactor), col="red")+ geom_point(aes(y=Wind * scaleFactor)) +
  scale_y_continuous(name="Solar.R,Ozone,Temp", sec.axis=sec_axis(~./scaleFactor, name="Wind")) +
  theme(
    axis.title.y.left=element_text(color="blue"),
    axis.text.y.left=element_text(color="blue"),
    axis.title.y.right=element_text(color="red"),
    axis.text.y.right=element_text(color="red")
  ) +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month",title = "Air Quality Over days on two scale Y axis") 


ggplot(my_aq.m, aes(date, variable)) + geom_tile(aes(fill = rescale),colour = "white") + scale_fill_gradient(low = "white",high = "steelblue") +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month", y="Variable",title = "Air Quality Heat map") + theme(legend.title=element_blank()) 


ggplot(my_aq.m, aes(x=month(date), y=day(date) )) + geom_tile(aes(fill = rescale),colour = "white") +  facet_grid(~ year(date), space="free_x", scales="free_x", switch="x") +
  scale_y_discrete(breaks = my_aq.m$variable) + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Month", y="Variable",title = "Air Quality Heat map") + theme(legend.title=element_blank()) 

ggplot(my_aq) +
  geom_point(aes(x=Wind,y=Temp,size=Ozone,color=Solar.R)) + theme(legend.position = "bottom",axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Wind", y="Ozone",title = "Air Quality Scatter Plot") 
