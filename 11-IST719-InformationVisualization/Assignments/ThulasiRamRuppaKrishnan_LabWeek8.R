###########################################################################
#
# Author : Ram Krishnan
# Purpose: Week 8
# "Class Social Network Data (structure and Cleaning)"
# uses LINKS-421-719Networks.csv
#      Nodes-421-719Network.csv
###########################################################################

library(igraph)

my.dir <- "C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST719\\Data\\"
link.data <- read.csv(paste0(my.dir,"links-421-719network.xls")
                      ,header = TRUE, stringsAsFactors = FALSE)

node.data <- read.csv(paste0(my.dir,"nodes-421-719network.xls")
                      ,header = TRUE, stringsAsFactors = FALSE)


colnames(link.data) <- gsub("\\.","",colnames(link.data))

link.data$X <- gsub(" |-","",link.data$X)

cbind(link.data$X,colnames(link.data)[-1])


node.data$Name <-  gsub(" |-","",node.data$Name)


cbind(node.data$Name,link.data$X)

M <- as.matrix(link.data[,-1])

rownames(M) <- colnames(M)
dim(M)

any(is.na(M))

M[is.na(M)]  <- 0

m[M>1]


g <- graph_from_adjacency_matrix(M)


##############################################################################################################
#
#               The graph object and the first plot
#
##############################################################################################################



vcount(g)
ecount(g)

plot.igraph(g)


g <- simplify(g)

plot.igraph(g)
par(mar=c(0,0,0,0))

plot.igraph(g,edge.arrow.size=0, edge.arrow.width=0)

E(g)$arrow.size <- 0
E(g)$arrow.width <- 0

plot.igraph(g)

g


V(g)$color <- "gold"
V(g)$frame.color <- "white"
V(g)$label.color <- "black"
E(g)$color <- "cadetblue"
V(g)$size <- 5

plot.igraph(g)


?igraph.plotting()

E(g)$curved <- .4


###################################################################################
#
#      Visualizing Centrality and Centrality measurement
#
###################################################################################



plot(degree(g))

par(mar= c(3,10,1,1))
barplot(sort(degree(g)), horiz= T, las = 2, main= "Number of Social Connections by Individual")

V(g)$degree <- degree(g)

V(g)$deg.out <- degree(g, mode="out")
V(g)$deg.in <- degree(g, mode="in")



barplot(V(g)$deg.out, horiz= T, las = 2
        , names.arg = V(g)$name
        , main= "Most friendliness degree by Indiviudal")



barplot(V(g)$deg.in, horiz= T, las = 2
        , names.arg = V(g)$name
        , main= "Most Important degree by Indiviudal")


#g.bak <- g
#g <- as.undirected(g)
#g <- g.bak


V(g)$close <- closeness(g, normalized = T, mode = "all")
V(g)$bet <- betweenness(g,directed = F)

library(plotrix)
my.pallet <- colorRampPalette(c("steelblue1","violet","tomato","red","red"))

V(g)$color <- rev(my.pallet(200))[round(1+rescale(V(g)$close,c(1,199)),0)]


plot.igraph(g)

V(g)$size <- 2 + rescale(V(g)$degree, c(0,13))
V(g)$label.cex <- .7+ rescale(V(g)$bet,c(0,1.25))


##########################################################################
#
#                Visualizing Social Network Structures
#
##########################################################################


cbind(V(g)$name, node.data$Name)

V(g)$class <- node.data$Class
V(g)$country <- node.data$Country
V(g)$year <- node.data$year


g <- delete_vertices(g,"JoHunter")
plot.igraph(g)

V(g)$shape <- "circle"
V(g)$shape[V(g)$class =="Wednesday"] <- "square"
V(g)$shape[V(g)$class =="Both"] <- "rectangle"

plot.igraph(g)


V(g)$color <- "gold"
V(g)$color[V(g)$country=="India"] <- "springgreen4"
V(g)$color[V(g)$country=="China"] <- "red"
V(g)$color[V(g)$country=="Both"] <- "purple"

plot.igraph(g)


V(g)$label.color <- "blue"
V(g)$label.color[V(g)$year==1] <- "black"

plot.igraph(g)


fc <- cluster_fast_greedy(as.undirected(g))
print(modularity(fc))

membership(fc)
V(g)$cluster <- membership(fc)
length(fc)
sizes(fc)

par(mar=c(1,1,1,1))
plot_dendrogram(fc, palette = rainbow(7), main="Social Network Cluster Dendrogram")

##################################################################################
#
# Visualizing Social Network Structures
#       use ist719NetworkObjects.rda
#
##################################################################################


my.dir <- "C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST719\\Data\\"
load(paste0(my.dir,"ist719networkobject.rda"))
par(mar = c(0,0,0,0))
plot.igraph(g)

l <- layout_in_circle(g)

V(g)$x <- l[,1]
V(g)$y <- l[,2]

plot.igraph(g)


l <- layout_with_fr(g)


V(g)$x <- l[,1]
V(g)$y <- l[,2]

plot.igraph(g)

l <- layout_as_star(g, center = "LeelaDeshmukh")
V(g)$x <- l[,1]
V(g)$y <- l[,2]
plot.igraph(g)


E(g)$color <- "gray"
E(g)[from("LeelaDeshmukh")]$color <- "red"

l <- layout_as_star(g, center = "LeelaDeshmukh")
plot.igraph(g)


l <- layout_with_kk(g)
V(g)$x <- l[,1]
V(g)$y <- l[,2]
plot.igraph(g)

















