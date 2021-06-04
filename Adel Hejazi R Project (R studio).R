#loading csv data to dataframe 

setwd("C:\\Users\\adeld\\OneDrive\\Desktop")
getwd()
install.packages("readxl")
library(readxl)
df<-read.csv(file.choose())
# "KSI_CLEAN.csv" this is the data name "killed or seriously injured"



df[df=='']<-NA #converting Null to Na  
#Getting familiar with data
dim(df)
colnames(df)
str(df)
summary(df)

# Checking the head of the dataset

head(df)
head(df,10)

# Checking the tail of the dataset
tail(df)

# number of missing values:
colSums(is.na(df))
#Percentage of missing values:
round(colMeans(is.na(df))*100,2)#2:digit=2

#data cleansing is 1)Handling duplicate data 2)Handling Missing Values 3)Handling outliers
#make a copy
df_org<-df

#Duplicate Data
/////////////////////////////////////////////////////////////////
duplicated(df)
sum(duplicated(df))
#returns for TRUE for observation that is duplicated otherwise returns FALSE
#Note: an observation considered duplicated if values of all features are exactly the same as another observation

#How many duplicated data are there?
sum(duplicated((df)))#--> there are 446 duplicated data that I will drop at the beginning
r1<-which(duplicated(df))
df<-df[-r1,]

#make a copy 
df_org<-df

# number of missing values:
colSums(is.na(df))
#Percentage of missing values:
round(colMeans(is.na(df))*100,2)#2:digit=2

tail(df)

#checking columns 
names(df)
#or
colnames(df)
str(df)

# Getting the summary of Data
summary(df)



#Q1.what is the distribution of (AUTOMOBILE)?
#how many missing values we have for loan.Status
sum(is.na(df$AUTOMOBILE))
#there is no missing value 

summary(df$AUTOMOBILE)
hist(df$AUTOMOBILE, breaks = 3, main = "Fatal",col="BROWN",xlab="AUTOMOBILE",ylab="Frequency")

ggplot

#Q2.what is the distribution of (VEHTYPE)?
#how many missing values we have for VEHTYPE
sum(is.na(df$VEHTYPE))

#we don't have  any missing value for VEHTYPE

#since VEHTYPE is categorical variable, in univaraite Analysis for summarization we will find frequency and for visualization simple  barplot 

counts <- table(df$VEHTYPE)
counts
barplot(c(count[2], count[3],count[4],count[5]), main="VEHTYPE",
        ylab="Number",col = 'blue',horiz = FALSE)

counts <- table(df$VEHTYPE)
counts 
barplot(counts, main="vEHTYPE",
        xlab="Number of VEH",col="blue",
        names.arg=c("Automobile","station wagon", "Bicycle Bus", "other",'construction Equipment'))

counts <- table(df$VEHTYPE)
barplot(counts, main="vEHTYPE", horiz=TRUE,
        names.arg=c("Automobile","station wagon", "Bicycle Bus", "other",'construction Equipment'),
        col="blue")



#Q3.Is there any relation between Accident, VISIBILTY and  (LIGHT)?
#bivariate analysis for VISIBILTY  Vs. LIGHT
#bivariate analysis for categorical vs. categorical:for visualization:stacked barchart or grouped bar chart, ...
#                                                   for summarization: Contingency table(two-way table)
                                                    #for test of independence: chi-square test

tbl<-table(df$VISIBILITY,df$LIGHT)
tbl


counts <- table(df$VISIBILITY, df$LIGHT)#you need 2 rows
counts
barplot(counts, main="Accidents by low VISIBILTY and LIGHT ",
        xlab="Number of accidents", ylab= "FREQUENCY",col=c("darkblue","red"),
        legend =  rownames(counts),args.legend = list(x="topright",bty="n",inset=c(0.4,0)))

#The Chi-square test statistic can be used if the following conditions are satisfied:
#1. N, the total frequency, should be reasonably large, say greater than 50.
#2. The sample observations should be independent. This implies that no individual item should be included twice or more in the sample.
#3. No expected frequencies should be small. Small is a relative term. Preferably each expected frequencies should be larger than 10 but in any case not less than 5.
install.packages("MASS") # we need to install MASS package 
library(MASS) 



tbl <- table(df$VISIBILITY,df$LIGHT)
tbl
tbl <-tbl[2:3,2:3]          # the contingency table
tbl
chisq.test(tbl)

#Test the hypothesis whether the VISIBILTY AND LIGHT are  independent at .05 significance level.
# we reject the NULL hypothesis, as the  accidents by low visibilty and light are statistically significantly associated
# p value is less then .05 significant level 


# Mosaic plots provide a way to visualize contingency tables.A mosaic plot is a visual representation of the association between two variables.
install.packages("vcd") #we need to install.packages
library(vcd)

library(vcd)
mosaic(tbl, shade=TRUE) 



#Q4.what is the distribution of MANOEVER. and is there any relation between MANOEVER (DRIVEACT)?

levels(df$MANOEUVER)
table(df$MANOEUVER)
addmargins(xtabs(~ MANOEUVER+DRIVACT,data=df))
prop.table(xtabs(~ MANOEUVER+DRIVACT,data=df ))


tbl <-table(df$MANOEUVER,df$DRIVACT)
tbl
tbl<-tbl <-tbl[2:4,2:4]                  # the contingency table
tbl


chisq.test(tbl)

# p value is more than .05 significant value, so we fail to reject the NULL hypothesis, THUS MANOVUVER AND DRIVACT are not indpependted, but they are depndent 

#Q5.what is DAY distribution?

summary(df$DAY)


#histogram
hist(df$DAY, breaks = 10, main = "ACCIDENTS",col="BROWN",xlab="DAY",ylab="Frequency")



#Q6. Is there any relationship between Hour  and  Injury ?

#Continouse Vs. Categorical  : For summaraization: group by categorical column an aggregate for numerical column
#                              For visualization: Grouped box plot,...
#                              For test of independence :1) if categorical column has only two levels :t-test
#                                                        2) if categorical column has more than two levels: ANOVA


agg1 <- aggregate(HOUR~ INJURY, df , mean)
agg1
names(agg1) <- c("INJURY","mean OF HOUR")
agg1

agg2<- cbind(aggregate(HOUR ~ INJURY, df , min),
             aggregate(HOUR~ INJURY, df , max)[,2],
             aggregate(HOUR~ INJURY, df , mean)[,2])

names(agg2) <- c("INJURY","min_HOUR","max_HOUR","mean_HOUR")
agg2

install.packages("ggplot2")
library(ggplot2)
qplot(INJURY, HOUR, data = df, 
      geom="boxplot", fill = INJURY)

# Changing histogram plot fill colors by INJURY and usinging semi-transparent fill
p<-ggplot(df, aes(x=HOUR, fill=INJURY, color=INJURY)) +
  geom_histogram(position="identity", alpha=0.5)
p

#add means lines 
library(plyr)

mu <- ddply(df, "INJURY", summarise, grp.mean=mean(HOUR,na.rm=T))
head(mu)
p<-p+geom_vline(data=mu, aes(xintercept=grp.mean, color=INJURY),
                linetype="dashed")
p


#Add density
p<-ggplot(df, aes(x=HOUR, fill=INJURY, color=INJURY)) +
  geom_histogram(aes(y=..density..),position="identity", alpha=0.5)+
  geom_density(alpha=0.6)
p
# Add mean lines and Change the legend position
p+geom_vline(data=mu, aes(xintercept=grp.mean, color=INJURY),
             linetype="dashed")+ theme(legend.position="top")+labs(title="HOUR histogram plot",x="HOUR", y = "Density")


#Q7.what is the distribution of speeding 

summary(df$SPEEDING)


#histogram

hist(df$SPEEDING, breaks = 3, main = "ACCIDENTS",col="BROWN",xlab="speeing",ylab="Frequency")

#Q8. what is the distrubution of RDSFCOND


tbl<-table(df$RDSFCOND)
tbl
tbl<-tbl[2:3]
tbl

# dot notation

#Q9 we need to find the summary for all catagorical columns with one mean of one numerical, in this question we want to find the mean of MONTH associate with all 
#catagorical columns 

#load library # we need to install packages doby

library(doBy)
## Using '.' on the right hand side of a formula means to stratify by
## all variables not used elsewhere:

summaryBy(MONTH~ . , df,na.rm=T)

#Q10. What is the distrbution of WEEKDAY and is there any association between WEEKDAY and MINUTES?

df$WEEKDAY
df$MINUTES

library(ggplot2)

#Generic X-Y Plotting (scatter plot)
plot(x = df$WEEKDAY, y = df$MINUTES,
     pch = 25, #Use the pch= option to specify symbols to use when plotting points, click on it and press F1 to see more options
     xlab = "WEEKDAY", ylab = "MINUTES", col = "brown")#col=color:You can specify colors in R by index, name, hexadecimal, or RGB
colors()#Returns the built-in color names 
#WE can right click on graph and save as or 
#or select plots tab then Exports
#or we can directly like following code save it


