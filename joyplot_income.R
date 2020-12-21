setwd("/media/xu/Elements/Study/HuMNetLab/Projects/TravelBehavior/")

library(ggplot2)
library(ggridges)

#theme_set(theme_ridges())
library(RColorBrewer)

dataPath = "/media/xu/Elements/Study/HuMNetLab/Data/Dallas/userData/"

routeData <- read.csv2(paste(dataPath, "userHomeLocationIncomeRoutes.csv", sep=''), sep=',')

routeData <- subset(routeData, numRoutes >= 1)

routeData['Routes'] <- lapply(routeData['numRoutes'], as.factor)
routeData['income'] <- lapply(routeData['income'], as.character)
routeData['income'] <- lapply(routeData['income'], as.numeric)

for (row in 1:nrow(routeData)) {
  numR <- routeData[row, "numRoutes"]
  if (numR>=4){
    route <- 4
    routeData[row, 'Routes'] = route
  }
}

# average income for each category
avg1 <- median(subset(routeData, Routes == 1, select = c('income'))$income)
avg2 <- median(subset(routeData, Routes == 2, select = c('income'))$income)
avg3 <- median(subset(routeData, Routes == 3, select = c('income'))$income)
avg4 <- median(subset(routeData, Routes == 4, select = c('income'))$income)

routeData <- routeData[order(routeData$Routes),]

average <- data.frame("avg" = c(avg1, avg1, avg2, avg2, avg3, avg3, avg4, avg4), 
                      "y" = c(1, 4, 1, 4, 1, 4, 1, 4), 
                      "grp" = c(1, 1, 2, 2, 3, 3, 4, 4))
average['grp'] <- lapply(average['grp'], as.factor)

Colormap<- colorRampPalette(rev(brewer.pal(11,'Spectral')))(32)
p <- ggplot(routeData, aes(x = `income`, y = `Routes`)) +
  geom_density_ridges_gradient(aes(fill = ..x..), scale = 3, size = 0.3 ) +
  scale_fill_gradientn(colours=Colormap, name = "Income [k$]") + 
  coord_cartesian(xlim = c(0, 100))
p <- p + theme_linedraw() + 
  theme(panel.grid.minor = element_line(colour = "#333333", size = 1.0)) + 
  scale_x_continuous(name="Income [k$]", minor_breaks = seq(0 , 100, 25), breaks = seq(0, 100, 25)) + 
  scale_y_discrete(name="# routes")
# p <- p + ggplot(average, aes(x=average$'avg', y=average$'y', group = average$'grp')) + 
#   geom_point() + geom_line()
p
ggsave(paste(dataPath, "routes_income_joyplot.pdf", sep=''),
       width = 5, height = 3)
