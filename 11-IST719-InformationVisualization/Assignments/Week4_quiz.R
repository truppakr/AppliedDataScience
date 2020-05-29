file_nm <- file.choose()

sales <- read.csv(file = file_nm, header = T
                  , stringsAsFactors = F)
# head(sales)

sales_2010 <- sales[sales$year=="2010",]
sales_2010_white <- sales_2010[sales_2010$type=="white",]
sales_2010_white_sum <- aggregate(sales_2010_white$units.sold,list(sales_2010_white$rep.region,sales_2010_white$sales.rep),sum)
sales_2010_white_max <- aggregate(sales_2010_white_sum$x,list(sales_2010_white_sum$Group.1),max)

sales_2010_white_max_rep <- merge(sales_2010_white_max,sales_2010_white_sum,by.x = c("Group.1","x"), by.y = c("Group.1","x"))
sales_2010_white_max_rep <- `colnames<-`(sales_2010_white_max_rep,c("rep.region","units.sold","sales.rep"))



sales_2012 <- sales[sales$year=="2012",]
# sales_2010_white <- sales_2010[sales_2010$type=="white",]
sales_2012_receipt_sum <- aggregate((sales_2012$units.sold*sales_2012$unit.price),list(sales_2012$rep.region,sales_2012$type),sum)
sales_2012_max <- aggregate(sales_2012_receipt_sum$x,list(sales_2012_receipt_sum$Group.1),max)

sales_2012_receipt_max <- merge(sales_2012_max,sales_2012_receipt_sum,by.x = c("Group.1","x"), by.y = c("Group.1","x"))
sales_2012_receipt_max <- `colnames<-`(sales_2012_receipt_max,c("rep.region","max.receipts","sales.type"))
sales_2012_receipt_max