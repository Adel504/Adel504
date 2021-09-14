##############################################################################################
#Changing directory
##############################################################################################
setwd('C:\\Users\\hmdra\\Desktop\\Metro College_Teaching Materials\\R\\my class')
getwd()






##############################################################################################
# loading csv data to dataframe 
##############################################################################################
df<-read.csv("credit_train.csv")
#train<-read.csv("credit_train.csv")

#or
#train<-read.csv(file.choose()) 


#test<-read.csv("credit_test.csv")

#train$source<-"train"
#test$source<-"test"
# smartbind the dataframes together
#library(plyr)
#df<-rbind.fill(train, test)#Combine data.frames by row, filling in missing columns



df[df=='']<-NA #converting Null to Na
##############################################################################################
#Getting fimiliar with data
##############################################################################################
dim(df)
colnames(df)
str(df)

# Checking the head of the dataset
head(df)
head(df,10)


# Checking the tail of the dataset
tail(df)


#I saw that the last observations are Na, I will drop them when I drop duplicated data in data cleansing part.

#data cleansing is 1)Handling duplicate data 2)Handling Missing Values 3)Handling outliers
#make a copy
df_org<-df
##########################################################################################
#Duplicate Data
#########################################################################################
duplicated(df)# returns for you TRUE for observation that is duplicated otherwise returns FALSE
#Note: an observation considered duplicated if values of all features are exactly the same as another observation

#How many duplicated data are there?
sum(duplicated((df)))#--> there are 10728 duplicated data that I will drop at the beginning
r1<-which(duplicated(df))
df<-df[-r1,]

#make a copy
df_org1<-df
#df<-df_org1

#checking columns 
names(df)
#or
colnames(df)
str(df)

# Getting the summary of Data
summary(df)


#dropping features are only for identification and we don't have any knowlege to extract meaningfull features from them
df[,c("Loan.ID","Customer.ID")]<-NULL




#summary(df)


#what is the distribution of target(Loan.Status)?
#how many missing values we have for loan.Status
sum(is.na(df$Loan.Status))

#since we have  just only one missing value for target,I'll drop that observation.
r1<-which(is.na(df$Loan.Status))
r1
df<-df[-r1,]
#since target is categorical variable, in univaraite Analysis for summarization I will find frequency and for visualization I plot: pie chart or barchart 

tbl<-table(df$Loan.Status)
tbl
tbl<-tbl[2:3]
tbl
#or

tbl<-aggregate(df$Loan.Status,list(df$Loan.Status),length)
tbl


# Pie Chart with Percentages
count<-table(df$Loan.Status)
count
freq1 <- c(count[2], count[3])
lbls <- c("Charged Off", "Fully Paid")
pct <- round(freq1/sum(freq1)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(freq1,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Loan.Status")
#as you can see we have 75% fully paid and 25% charged off, so we are dealing with unbalaced data

# Simple Bar Plot
counts <- table(df$Loan.Status)
counts
barplot(c(count[2], count[3]), main="Loan Status",
        ylab="Number",col = 'blue',horiz = FALSE)


#you can repeat this process for all categorical columns for example for Term

#what is the distribution of Term?
#how many missing values we have for Term
sum(is.na(df$Term))

#we don't have  any missing value for Term

#since Term is categorical variable, in univaraite Analysis for summarization I will find frequency and for visualization I plot: pie chart or barchart 

tbl<-table(df$Term)
tbl
tbl<-tbl[2:3]
tbl



# Pie Chart with Percentages
count<-table(df$Term)
count
freq1 <- c(count[2], count[3])
lbls <- c("Charged Off", "Fully Paid")
pct <- round(freq1/sum(freq1)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(freq1,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Term")

# Simple Bar Plot
counts <- table(df$Term)
counts
barplot(c(count[2], count[3]), main="Term",
        ylab="Number",col = 'blue',horiz = FALSE)

#what is the relation between Term and target(Loan.Status)?
#bivariate analysis for Term Vs. Loan.statuse
#bivariate analysis for categorical vs. categorical:for visualization:stacked barchart or grouped bar chart, ...
#                                                   for summarization: Contingency table(two-way table)
#                                                   for test of independence: chi-square test


# Stacked Bar Plot with Colors and Legend
tbl<-table(df$Term,df$Loan.Status)
tbl
counts<-tbl[2:3,2:3]
counts
barplot(counts, main="Term Vs. Loan Status",
        xlab="Loan Status", col=c("darkblue","red"),
        legend = rownames(counts))


add<-addmargins(xtabs(~ Term+Loan.Status,data=df))
add
add[2:4,2:4]
prop.table(xtabs(~ Term+Loan.Status,data=df))[2:3,2:3]

##########################################################################################

#Chi-squared Test of Independence(Pearson's Chi-squared)

##########################################################################################
#The chi-square test of independence is used to analyze the frequency table (i.e. contengency table) formed by
#two categorical variables. The chi-square test evaluates whether there is a significant association between the
#categories of the two variables.In other word Chi-Square test is a statistical method which used to determine if 
#two categorical variables have a significant correlation between them.


#The Chi-square test statistic can be used if the following conditions are satisfied:
#1. N, the total frequency, should be reasonably large, say greater than 50.
#2. The sample observations should be independent. This implies that no individual item should be included twice or more in the sample.
#3. No expected frequencies should be small. Small is a relative term. Preferably each expected frequencies should be larger than 10 but in any case not less than 5.

library(MASS)       # load the MASS package
#Problem:
#Test the hypothesis whether the Loan.Status is independent of the Term  at .05 significance level.
# Null hypothesis  Loan status is independent of Term of loan
#Solution
#We apply the chisq.test function to the contingency table tbl, and found the p-value to be 2.2e-16
tbl <- table(df$Loan.Status,df$Term)
tbl
tbl <-tbl[2:3,2:3]              # the contingency table
tbl
chisq.test(tbl)
# or Mosaic plots provide a way to visualize contingency tables.A mosaic plot is a visual representation of the association between two variables.
library(vcd)
mosaic(tbl, shade=TRUE) 

#or
#Association Plots
assoc(tbl, shade=TRUE)
#Answer:
#As the p-value 2.2e-16 is less than the .05 significance level, we reject the null hypothesis that
#the Loan.Status is independent of the Term and conclude that in our data, the Loan.Status and the Term are statistically significantly associated (p-value = 0)

#what is the relation between Years.in.current.job and target(Loan.Status)?
#levels(df$Years.in.current.job)
table(df$Years.in.current.job)
addmargins(xtabs(~ Years.in.current.job+Loan.Status,data=df))
prop.table(xtabs(~ Years.in.current.job+Loan.Status,data=df))

#Problem:
#Test the hypothesis whether the Loan.Status is independent of the Years.in.current.job  at .05 significance level.
# Null hypothesis  Loan status is independent of  Years.in.current.job


#Solution
#We apply the chisq.test function to the contingency table tbl, and found the p-value to be 2.2e-16
tbl <-table(df$Loan.Status,df$Years.in.current.job)
tbl
tbl<-tbl [-1,-1]                # the contingency table
chisq.test(tbl)
#Answer:
#As the p-value 2.2e-16 is less than the .05 significance level, we reject the null hypothesis that
#the Years.in.current.job is independent of the Term so there is association between loan status
# and years in current job at 5% significant level


#5)what is the relation between Home.Ownership and target(Loan.Status)?
table(df$Home.Ownership)
df$Home.Ownership[df$Home.Ownership=="HaveMortgage"]<-"Home Mortgage"
addmargins(xtabs(~ Home.Ownership+Loan.Status,data=df))
prop.table(xtabs(~ Home.Ownership+Loan.Status,data=df))

#Problem:
#Test the hypothesis whether the Loan.Status is independent of the Home.Ownership at .05 significance level.
# Null hypothesis  Loan status is independent of  Home Ownership

#Solution
#We apply the chisq.test function to the contingency table tbl, and found the p-value to be 2.2e-16
tbl <- table(df$Loan.Status,df$Home.Ownership)
tbl
tbl <-tbl[-1,-c(1,2)]
tbl# the contingency table
chisq.test(tbl)
#Answer:
#As the p-value 2.2e-16 is less than the .05 significance level, we reject the null hypothesis that
#the Loan.Status is independent of the Home.Ownership
#so the Loan.Status and Home ownership are statistically significantly associated (p-value = 0)



#Feature Engineering 
#1)Find and Replace (for numeric variables some function created and applied)(For example converting Other to other or replacing 'HaveMortgage' with 'Home Mortgage') 
#2)segmentation
#3)Encoding
#3-1) Mapping with numbers for ordinal categorical variables
#3-2) One hot encoding (convert categorical variable into dummy): Create dummy variables from categorical variable that are nominal (not ordinal) categorical variable. It's also a good practice to drop the first one to avoid linear dependency between the resulted features.

#4)Create new columns or PCA (Principal Component Analysis)
#I will create RMI (ratio of Monthly Debt to Annual Income) and RMC (ratio of Monthly Debt to Maximum Open Credit) after cleaning data for test and train datasets

#5)Scaling: I used StandardScaler to scale my data.



#what is the relation between Purpose and target(Loan.Status)?
table(df$Purpose)
df$Purpose[df$Purpose=="Other"]<-"other"
table(df$Purpose)

addmargins(xtabs(~ Purpose+Loan.Status,data=df))
prop.table(xtabs(~ Purpose+Loan.Status,data=df))


tbl <-table(df$Loan.Status,df$Purpose)
tbl
tbl<-tbl[-1,-c(1,12)]                 # the contingency table
tbl
#what is the relation between Loan status and Purpose?

chisq.test(tbl)
#Chi-squared approximation may be incorrect because you have a group will small frequency
#Hint: when you have lots off levels in one column you can try segmentation

df$Purpose[df$Purpose=="renewable_energy"]<-"other"
tbl <-table(df$Loan.Status,df$Purpose)
tbl
tbl<-tbl[-1,-c(1,12,13)]                 # the contingency table
tbl

chisq.test(tbl)
#Answer:
#As the p-value 2.2e-16 is less than the .05 significance level, we reject the null hypothesis that
#the Loan.Status is independent of the purpose of loan
#so the Loan Status and the purpose of loan are statistically significantly associated (p-value = 0)


#what is the relation between Home.Ownership and Purpose?
addmargins(xtabs(~ Home.Ownership+Purpose,data=df))
prop.table(xtabs(~ Home.Ownership+Purpose,,data=df))
tbl = table(df$Home.Ownership,df$Purpose)
tbl
tbl[-c(1,2),-c(1,12,13)]                 # the contingency table
tbl



#make a copy
df_orginal<-df
#df<-df_orginal

#credit score



#what is Credit.Score distribution?

summary(df$Credit.Score)

#histogram
hist(df$Credit.Score, breaks = 5, main = "Credit.Score",col="blue",xlab="Credit.Score",ylab="Frequency")


# Boxplot of Credit.Score by Loan.Status
boxplot(Credit.Score ~ Loan.Status,data=df, main="Credit.Score",
        xlab="Loan.Status", ylab="Credit.Score",col="blue")

summary(df["Credit.Score"])
#If you notice the maximum of credit scoe is 7510 which is strange considering the credit score are within the range of 300-850.
dfc<-df[which(df["Credit.Score"]>850),]
dfc$Credit.Score
summary(dfc["Credit.Score"])
nrow(dfc)

#it looks like some of the credit score are just scaled up by 10. let me make sure
count<-0
for (val in dfc$Credit.Score){
  if (val%%10 !=0) {count=count+1}
}


count#--> count = 0 means all the credit scores greater than 850 are scaled up by 10 



df$Credit.Score<-ifelse(df$Credit.Score>850, df$Credit.Score/10, df$Credit.Score)

summary(df["Credit.Score"])


agg1 <- aggregate(Credit.Score~ Loan.Status, df , mean)
names(agg1) <- c("Loan Status","mean of Credit Score")
agg1

agg2<- cbind(aggregate(Credit.Score ~ Loan.Status, df , min),
             aggregate(Credit.Score~ Loan.Status, df , max)[,2],
             aggregate(Credit.Score~ Loan.Status, df , mean)[,2])

names(agg2) <- c("Loan.Status","min_Credit.Score","max_Credit.Score","mean_Credit.Score")
agg2


#Segmentation for credit score
#credit scores from 300 to 560 are considered Poor, credit scores from 560 to 660 are considered Fair, credit scores from 660 to 724 are considered good; 725 to 759 are considered very good; and 760 and up are considered excellent.

##########################################################################################

#discritization

##########################################################################################
#make a copy
df_orginal2<-df
#df<-df_orginal2
#discritizing of Credit.Score
names(df)
cat1<-rep(NA,nrow(df))
df<-cbind(df[,1:4],cat1,df[,5:17])

df$cat1<-df$Credit.Score
df$cat1[300<=df$cat1 & df$cat1<=559]<-"Poor"
df$cat1[560<=df$cat1 & df$cat1<=659]<-"Fair"
df$cat1[660<=df$cat1 & df$cat1<=724]<-"Good"
df$cat1[725<=df$cat1 & df$cat1<=759]<-"Very Good"
df$cat1[760<=df$cat1 & df$cat1<=850]<-"Excellent"
names(df)[5]<-"Credit.Score.Status"
names(df)
head(df)


#What is  the credit score  status distribution and is there any association between loan status and credit score status at 5% significant level?
sum(is.na(df$Credit.Score.Status))
r1<-which(is.na(df$Credit.Score.Status))
r2<-which(is.na(df$Loan.Status))
dfcat<-df[-c(r2,r2),]

table(dfcat$Credit.Score.Status)

count<-table(dfcat$Credit.Score.Status)
count

# Pie Chart with Percentages
slices <- c(count[1], count[2], count[3])
lbls <- c("Fair", "Good", "Very Good ")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(slices,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Credit Score Status")


add<-addmargins(xtabs(~ Credit.Score.Status+Loan.Status,data=dfcat))
add
add[,-1]
pro<-prop.table(xtabs(~ Credit.Score.Status+Loan.Status,data=df))
pro
pro[,-1]

#Problem:
#Test the hypothesis whether the Loan.Status is independent of the credit score status at .05 significance level.
# Null hypothesis  Loan status is independent of  credit score status 

#Solution
#We apply the chisq.test function to the contingency table tbl, and found the p-value to be 2.2e-16
tbl <- table(dfcat$Loan.Status,dfcat$Credit.Score.Status)
tbl
tbl <-tbl[-1,]
tbl# the contingency table
chisq.test(tbl)
#Answer:
#As the p-value 2.2e-16 is less than the .05 significance level, we reject the null hypothesis that
#the Loan.Status is independent of the Credit.Score.Status
#so the Loan.Status and Credit.Score.Status are statistically significantly associated (p-value = 0)
assoc(tbl, shade=TRUE)




#what is the relation between Current.Loan.Amount and target(Loan.Status)?

summary(df$Current.Loan.Amount)
df$Current.Loan.Amount
hist(df$Current.Loan.Amount, breaks = 5, main = "Current.Loan.Amount",col="blue")


# Boxplot of Current.Loan.Amount by Loan.Status
boxplot(Current.Loan.Amount ~ Loan.Status,data=df, main="Current.Loan.Amount",
        xlab="Loan.Status", ylab="Current.Loan.Amount",col="blue")

#it looks like instead of Na  for Current.Loan.Amount in some observation we have 99999999.Let me fix it first.
dfc2<-df[which(df["Current.Loan.Amount"]==99999999),]

nrow(dfc2)


df$Current.Loan.Amount<-ifelse(df$Current.Loan.Amount==99999999, NA, df$Current.Loan.Amount)

summary(df$Current.Loan.Amount)
boxplot(Current.Loan.Amount ~ Loan.Status,data=df, main="Current.Loan.Amount",
        xlab="Loan.Status", ylab="Current.Loan.Amount",col="blue")
