
names(marvel_dc) <- make.names(names(marvel_dc))
summary(marvel_dc)
scatterplot(APPEARANCES~page_id | group, regLine=FALSE, smooth=FALSE, 
  boxplots=FALSE, by.groups=TRUE, data=marvel_dc)
library(rgl, pos=67)
library(nlme, pos=68)
library(mgcv, pos=68)
scatter3d(APPEARANCES~page_id+YEAR|group, data=marvel_dc, surface=FALSE, 
  residuals=TRUE, parallel=FALSE, bg="white", axis.scales=TRUE, grid=TRUE, 
  ellipsoid=FALSE)
GLM.2 <- glm(group ~ YEAR +GSM + EYE + HAIR, family=binomial(logit), 
  data=marvel_dc)
summary(GLM.2)
exp(coef(GLM.2))  # Exponentiated coefficients ("odds ratios")
library(lattice, pos=70)
Boxplot( ~ YEAR, data=marvel_dc, id=list(method="y"))
load("C:/Users/rkrishnan/Documents/01 Personal/MS/SCM 651/week 7/Business+Analytics+-+Week+7+oj.csv")
fix(marvel_dc)
data()
Orange <- 
  read.table("C:/Users/rkrishnan/Documents/01 Personal/MS/SCM 651/week 7/Business+Analytics+-+Week+7+oj.csv",
   header=TRUE, sep="", na.strings="NA", dec=".", strip.white=TRUE)
summary(Orange)
load("C:/Users/rkrishnan/Documents/01 Personal/MS/SCM 651/week 7/Business+Analytics+-+Week+7+oj.csv")
fix(Orange)
Dataset <- 
  read.table("C:/Users/rkrishnan/Documents/01 Personal/MS/SCM 651/week 7/Business+Analytics+-+Week+7+oj.csv",
   header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
Boxplot( ~ price, data=Dataset, id=list(method="y"))
with(Dataset, Barplot(brand, xlab="brand", ylab="Frequency"))

