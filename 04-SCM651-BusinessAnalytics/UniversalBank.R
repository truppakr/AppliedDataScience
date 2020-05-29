
UniveralBank <- 
  read.table("C:/Users/rkrishnan/Documents/01 Personal/MS/SCM 651/week 8/scm651_homework_4_universal_bank.csv",
   header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
scatterplot(Age~PersonalLoan, regLine=FALSE, smooth=FALSE, boxplots=FALSE, 
  data=UniveralBank)
scatterplot(Income~PersonalLoan, regLine=FALSE, smooth=FALSE, 
  boxplots=FALSE, data=UniveralBank)
scatterplot(PersonalLoan~Income, regLine=FALSE, smooth=FALSE, 
  boxplots=FALSE, data=UniveralBank)
scatterplot(PersonalLoan~Income, regLine=FALSE, smooth=list(span=0.5, 
  spread=FALSE), boxplots=FALSE, data=UniveralBank)
GLM.Logit.UniversalBank <- glm(PersonalLoan ~ Income + Age + Family + CCAvg,
   family=binomial(logit), data=UniveralBank)
summary(GLM.Logit.UniversalBank)
exp(coef(GLM.Logit.UniversalBank))  
  # Exponentiated coefficients ("odds ratios")

