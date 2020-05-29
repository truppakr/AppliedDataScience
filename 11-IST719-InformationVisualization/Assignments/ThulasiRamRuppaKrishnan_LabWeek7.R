
##################################################
#
# Author : Ram Krishnan
# Purpose: Week 7 Lab:: Maps
# Uses: MapLectureData.csv
#
##################################################


file_dir <- "C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST719\\Week7\\"

df <- read.csv(file= paste0(file_dir,"maplecturedata.xls")
               , header = T
               , stringsAsFactors = F)
head(df)


plot(df$x,df$y)

polygon(df$x,df$y,col = "firebrick1",border = NA)

library(maps)
library(mapproj)

map(database = "world")

map("world", regions = "india")
map("world", regions = "China")

map("world", regions = c("India","Pakistan"),fill = TRUE, col= c("Orange","Brown"))
map("world", regions = "Finland")


m <- map("state")
plot(m$x,m$y)


map("state",fill = TRUE, col = c("Orange","red","yellow"))


map("county",region = "New York", fill = T, col = terrain.colors(20))

library(rnaturalearth)
library(rnaturalearth)
# install.packages("rnaturalearth")


india <- ne_states(country = "india")

map(india)
india

# memory.size()
# gc()
# memory.size()
# dev.off()
# memory.size()
# object.size(sales)

# install.packages("backports")
# devtools::install_github("ropensci/rnaturalearthhires")


attributes(india)
names(india)

india$name
map(india, namefield ="name", region = c("Gujarat","Rajasthan","Madhya Pradesh") 
    , fill = TRUE
    , col = c("Orangered","white","springgreen4"))

# install.packages("raster")

library(raster)


india <- raster::getData("GADM",country ="IND", level =1)
map(india)

india$NAME_1

map(india, namefield = "NAME_1", region = "Gujarat")


india <- raster::getData("GADM",country ="IND", level =2)
map(india)

india$NAME_2

map(india, namefield = "NAME_2", region = "North 24 parganas"
    , fill = TRUE
    , col = "springgreen4")



china <- raster::getData("GADM",country ="CHN", level =2)
map(china)



############################################################################################
#
#   Choropleth map problem
#
############################################################################################
# thematic maps, color maps to


my.dir <- "C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST719\\Week7\\"

fname <- paste0(my.dir,"shootings.rda")
load(fname)

# Which state has the most mass-shooting victims in the US?

View(shootings)

shootings$Total.Number.of.Victims

sort(shootings$State)

tmp.vec <- gsub("^\\s+|\\s+$","",shootings$State)
sort(tmp.vec)

shootings$State <- tmp.vec

agg.dat <- aggregate(shootings$Total.Number.of.Victims, list(shootings$State),sum)

colnames(agg.dat) <- c("state","victims")


num.cols <- 10
my.color.vec <- rev(heat.colors(num.cols))

pie(rep(1,num.cols), col = my.color.vec)

agg.dat

library(plotrix)
agg.dat$index <- round(rescale(x=agg.dat$victims, c(1,num.cols)))
agg.dat$color <- my.color.vec[agg.dat$index]



agg.dat

m <- map("state")

m$names

state.order <- match.map(database = "state", regions = agg.dat$state
                         , exact = FALSE, warn = TRUE)


cbind(m$names,agg.dat$state[state.order])

map("state", col=agg.dat$color[state.order], fill = TRUE
    , resolution = 0, lty=1, projection = "polyconic", border ="tan")
 



############################################################################################
#
#   points and geocoding
#   uses NewYorkLibraries.csv
#
############################################################################################

library(ggmap)

libs <- read.csv(paste0(my.dir,"NewYorkLibraries.xls")
                 , header = TRUE, quote = "\""
                 , stringsAsFactors = FALSE)



map("world")
points(0,0,col="red",cex=3,pch=8)
abline(h=43, col="blue", lty=3)
abline(v=-76, col="blue", lty=3)


us.cities
map("state")
my.cols <- rep(rgb(1,.6,.2,.7), length(us.cities$name))
my.cols[us.cities$capital>0] <- rgb(.2,.6,1,.9)

points(us.cities$long, us.cities$lat, col= my.cols
       , pch =16
       , cex = rescale(us.cities$pop,c(.5,7)))


geocode("3649 Erie Blvd East, Dewit, ny", source="dsk")
# https://github.com/location-iq/locationiq-r-client
# install.packages("jsonlite")
# install.packages("httr")
# install.packages("caTools")

library(locationiq)

# locationiq::Location


table(libs$CITY)

index <- which(libs$CITY %in% c("SYRACUSE","DEWITT","FAYETTEVILLE"))
addy <- paste(libs$ADDRESS[index],libs$CITY[index], libs$STABR[index], sep = ", ")


map("county","new york", fill = TRUE, col = "orange")

g.codes <- geocode(addy,source = "dsk")

# 43.0481° N, 76.1474° W
# 43.0427° N, 76.0678° W
# 43.0298° N, 76.0044° W

# "THE GALLERIES, 447 S. SALINA ST., SYRACUSE, NY"    43° 5' 20.2092, 76° 9' 16.1280''
# "SHOPPINGTOWN 3649 ERIE BLVD EAST, DEWITT, NY"  
# "406 CHAPEL DRIVE, SYRACUSE, NY"                 
# "300 ORCHARD STREET, FAYETTEVILLE, NY"          
# "4840 WEST SENECA TURNPIKE, SYRACUSE, NY"



points(c(-76.14,-76.10,-76.01,-76.05),c(43.041,43.042,43.043,43.044),col = "blue", cex = 1.1, pch = 16)



############################################################################################
#
#   Rworldmaps package
#   uses countries.csv
#
############################################################################################

library(rworldmap)
install.packages("rworldmap")
library(plotrix)

countries = read.delim(paste0(my.dir,"countries.xls")
                       , quote = "\""
                       , header = TRUE
                       , sep = ";"
                       , stringsAsFactors = FALSE)


range(countries$Life.expectancy)

# zap <- which(countries$Life.expectancy==0.0)
rm(zap)
countries <- countries[-zap,]

num.cat <- 10

iso3.codes <- tapply(countries$Country..en.
                     , 1:length(countries$Country..en.)
                     , rwmGetISO3)

df <- data.frame(country = iso3.codes, labels = countries$Country..en.
                 , life = countries$Life.expectancy)

df.map <- joinCountryData2Map(df, joinCode = "ISO3"
                              , nameJoinColumn = "country")

par(mar = c(0,0,1,0))

mapCountryData(df.map
               , nameColumnToPlot = "life"
               , numCats = num.cat
               , catMethod = 
                 c("pretty","fixedwidth","diverging","quantiles")[4]
               , colourPalette = colorRampPalette(
                 c("orangered","palegoldenrod","forestgreen")
               )(num.cat)
               , oceanCol = "royalblue4"
               , borderCol = "peachpuff4"
               , mapTitle = "Life Expectancy"
               
               )


############################################################################################
#
#   ggmaps package
#   uses IndiaReportedRapes.csv
#
############################################################################################


library(ggmap)
library(raster)

reported <- read.csv(paste0(my.dir, "IndiaReportedRapes.xls")
                     , header = TRUE, quote = "\""
                     , stringsAsFactors = FALSE)

india <- raster::getData("GADM", country ="IND", level=1)

cbind(unique(reported$Area_Name),india$NAME_1)

india$NAME_1[india$NAME_1 =="NCT of Delhi"] <- "Delhi"
india$NAME_1 <- gsub(" and ", " & ", india$NAME_1)

map <- fortify(india, region = "NAME_1")

head(map)

crimes <- aggregate(reported$Cases, list(reported$Area_Name), sum)

colnames(crimes) <- c("id","ReportedRapes")

crimes[order(crimes$ReportedRapes),]

my.map <- merge(x=map, y= crimes, by="id")

ggplot() + geom_map(data=my.map, map=my.map) +
  aes(x= long, y= lat, map_id= id, group = group
      , fill = ReportedRapes)+
  theme_minimal() + ggtitle("Reported Rapes in India")



# some requires Google API key, see ?register_google

## basic usage
########################################

# lon-lat vectors automatically use google:
?register_google

(map <- get_map(c(-97.14667, 31.5493)))
str(map)
ggmap(map)

# bounding boxes default to stamen
(map <- get_map(c(left = -97.1268, bottom = 31.536245, right = -97.099334, top = 31.559652)))
ggmap(map)

# characters default to google
(map <- get_map("orlando, florida"))
ggmap(map)


## basic usage
########################################

(map <- get_map(maptype = "roadmap"))
(map <- get_map(source = "osm"))
(map <- get_map(source = "stamen", maptype = "watercolor"))

map <- get_map(location = "texas", zoom = 6, source = "stamen")
ggmap(map, fullpage = TRUE)






############################################################################################
#
#   ggmaps package
#   uses bikes.rds
#        nyct2010_17a
#
############################################################################################



shape.dat.dir <- "C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST719\\Data\\"

library(stringr)
library(rgdal)
library(raster)
library(TeachingDemos) # for zoomplot

bikes <- readRDS(paste0(shape.dat.dir,"bikes.rds"))
nypp <- readOGR(paste0(shape.dat.dir,"nyct2010_17a")
                , "nyct2010"
                , stringsAsFactors = FALSE)


syr.neighbourhood <- readOGR(paste0(shape.dat.dir,"syracuse-neighborhoods_ny.geojson"))


par(mar= c(.5,.5,.5,.5))

plot(nypp,border ="bisque4", lwd =.5)
zoomplot(c(978000,999800), ylim = c(185000,225000))

df <- data.frame(lat = bikes$start.station.latitude,
                 lon = bikes$start.station.longitude)


head(df)


point.tab <- sort(table(paste(df$lat,df$lon)), decreasing =TRUE)

point.tab[1:3]






