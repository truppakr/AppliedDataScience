---
title: "Week 2 Homework Assignment"
output:
  pdf_document: default
  html_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r Assignment}
# IST687	- Writing	functions	and	doing	some	initial	data	analysis
myCars<-mtcars
myCars

# Step	1:	What	is	the	hp		(hp	stands	for	"horse	power")
# What	is	the	highest	hp?

maxHP <- max(myCars$hp)
maxHP

#Which	car	has	the	highest	hp?

CarWithHighHP <- row.names( myCars[which.max(myCars$hp),] )
CarWithHighHP

# Step	2:	Explore	mpg	(mpg	stands	for	"miles	per	gallon")		
# What	is	the	highest	mpg?

maxMPG <- max(myCars$mpg)
maxMPG

#Which	car	has	the	highest	mpg?
CarWithmaxMPG <- row.names( myCars[which.max(myCars$mpg),] )
CarWithmaxMPG

#Create	a	sorted	dataframe,	based	on	mpg

myCarsSorted <- myCars[order(myCars$mpg),]
myCarsSorted

#Step	3:	Which	car	has	the	"best"	combination	of	mpg	and	hp?
# Logic I used: calculate mpg per horsepower for each cars and the car with max miles per galon per orse power has got the best combination of hp and mpg
 
mpg_per_hp <- (myCars$mpg)/(myCars$hp)
mpg_per_hp
bestCar_hp_mpg <- row.names(myCars[which.max((myCars$mpg)/(myCars$hp)),]) 
bestCar_hp_mpg


# Step	4:	 Which	car	has "best"	car combination	of	mpg	and	hp,	where	mpg	and	hp	must	be	given	equal	weight

# we need to check the behavior of mpg and hp with repect to each other and to check that lets plot the data in a chart. I am using ggplot to see how their behaviour 

# Refernces from https://rpubs.com/BillB/217355
library(ggplot2)
library(ggrepel)

ggplot(myCars,aes(myCars$hp,myCars$mpg,label=row.names(myCars)))+ geom_point() +
         geom_smooth(method = "lm", se = FALSE)   + 
    geom_label_repel(aes(label = row.names(myCars)),
                     box.padding   = 0.35, 
                     point.padding = 0.5,
                     segment.color = 'grey50') +
    theme_classic()


# The plot shows that with increase in horsepower the mpg value drops. To select car with best combination of mpg and hp with equal weightage we need to get to the midpoint of this linear model and the cars that fall in the mid ranges are Ferrari Dino and Merc 450SL

```