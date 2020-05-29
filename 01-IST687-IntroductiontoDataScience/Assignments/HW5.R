
mv_URL <- "http://data.maryland.gov/api/views/pdvh-tf2u/rows.json?accessType=DOWNLOAD"

mvApiResult <- getURL(mv_URL)

namesOfColumns <-
  c("CASE_NUMBER","BARRACK","ACC_DATE","ACC_TIME","ACC_TIME_CODE","DAY_OF_WE
    EK","ROAD","INTERSECT_ROAD","DIST_FROM_INTERSECT","DIST_DIRECTION","CITY_NA
    ME","COUNTY_CODE","COUNTY_NAME","VEHICLE_COUNT","PROP_DEST","INJURY","COLLI
    SION_WITH_1","COLLISION_WITH_2")

mvResults <- fromJSON(mvApiResult)
summary(mvResults[1])
summary(mvResults[2])
mvData <-mvResults$data

nullToNA <- function(x) {
  x[sapply(x, is.null)] <- NA
  return(x)
}

lapply(mvData, nullToNA)
mv_df <- data.frame(matrix(unlist(lapply(mvData,nullToNA)),nrow=18638,ncol = 26,byrow = T), stringsAsFactors = FALSE)
mv_df <- mv_df[,-c(1:8)]
colnames(mv_df) <-c("CASE_NUMBER","BARRACK","ACC_DATE","ACC_TIME","ACC_TIME_CODE","DAY_OF_WEEK","ROAD","INTERSECT_ROAD","DIST_FROM_INTERSECT","DIST_DIRECTION","CITY_NAME","COUNTY_CODE","COUNTY_NAME","VEHICLE_COUNT","PROP_DEST","INJURY","COLLISION_WITH_1","COLLISION_WITH_2")


sqldf('select count(case_number) accidents_cnt from mv_df where trim(day_of_week) ="SUNDAY"')
sqldf('select count(1) accidents_with_injury from mv_df where injury="YES" ')
sqldf('select trim(day_of_week) day_of_week,count(1) injuries_cnt from mv_df where injury="YES" group by trim(day_of_week) order by case trim(day_of_week) when "SUNDAY" then 1 when "MONDAY" then 2 when "TUESDAY" then 3 when "WEDNESDAY" then 4 when "THURSDAY" then 5 when "FRIDAY" then 6 when "SATURDAY" then 7 end')


mv_df$DAY_OF_WEEK <-sapply(mv_df$DAY_OF_WEEK,trimws,which='right')
colnames<-`(matrix(tapply(mv_df$DAY_OF_WEEK, mv_df$DAY_OF_WEEK=='SUNDAY', length)[2]),"accidents_cnt")
colnames<-`(matrix(tapply(mv_df$CASE_NUMBER, mv_df$INJURY=='YES', length)[2]),"accidents_with_injury")
tapply(mv_df[which(mv_df$INJURY=='YES'),][,1], mv_df[which(mv_df$INJURY=='YES'),][,which(colnames(mv_df)=="DAY_OF_WEEK")]  , length)

