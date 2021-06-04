

-----SQL Project Phase2 ----- 10 questions 

1.	--Create a view to get all customers with checking account from ON province. [Moderate]

Create view VONCheking

as
select Customer.CustomerID,Customer.CustomerFirstName+' '+Customer.CustomerLastName [Customer Name],Customer.province,at.AccountTypeDescription
from Customer  
  join Customer_Account  
   on Customer.CustomerID=Customer_Account.CustomerID
   join Account 
   on Customer_Account.AccountID=Account.AccountID
   join AccountType at
   on Account.AccountTypeID=at.AccountTypeID
where at.AccountTypeDescription='Cheqing' and Customer.province='ON';

select * from VONCheking
go
use db_CanadaBank

go
select * from Account
select * from AccountStatusType
select * from AccountType
select * from Customer



go

2- --Create a view to get all customers with total account balance (including interest rate) greater than 5000. [Advanced]

select * from vw_customer_5000

create view vw_customer_5000
as
select Customer.customerID,Customer.customerfirstname,Customer.customerlastname,
Account.currentbalance,SavingsInterestRates.interestratevalue,(Account.currentbalance+(Account.currentbalance*SavingsInterestRates.interestratevalue))
as Balancewithinterestrate 
from Customer 
join Customer_Account  
on Customer_Account.CustomerID=Customer.CustomerID
join account 
on Account.AccountID=Customer_Account.AccountID
join SavingsInterestRates 
on SavingsInterestRates.InterestSavingsRateID=Account.InterestSavingsRateID
where (Account.CurrentBalance+(Account.CurrentBalance*SavingsInterestRates.InterestRateValue))>5000 
go

3.	--Create a view to get counts of checking and savings accounts by customer. [Moderate]

Create View checking_saving

as
select customer.CustomerFirstName+' '+ Customer.CustomerLastName as [customer Name],AccountType.AccountTypeDescription as[Account Type]
,count(Customer.customerID) [count of account]

from Customer


join Customer_Account on Customer.CustomerID=Customer_Account.CustomerID
join Account on Customer_Account.AccountID=Account.AccountID
join AccountType on Account.AccountTypeID=AccountType.AccountTypeID
group by Customer.CustomerFirstName,Customer.CustomerLastName,AccountType.AccountTypeDescription


select * from  checking_saving



go



4.---Create a view to get any particular user’s login and password using AccountId. [Moderate]

Create view withaccountID
as
select LA.accountID,Ul.userlogin,ul.userpassword
from Login_Account LA,userlogins ul
where la.userloginID=ul.UserLoginID

select * from withaccountID 

go

5.	---Create a view to get all customers’ overdraft amount. [Moderate]
create view overdraft 
as 
select CustomerID, OverDraftamount
from Customer join OverDraftLog
on Customer.AccountID = OverDraftLog.AccountID


 go

 create proc sp4update
as
update userlogins
set userlogin=concat('user_',userlogin);
go
exec sp4update
select * from UserLogins

go



6-	----Create a stored procedure to add “User_” as a prefix to everyone’s login (username). [Moderate]

        Create proc sp4update

		
		
		
as
update userlogins
set userlogin = concat('user_',userlogin);
go
exec sp4update

select * from UserLogins
use db_CanadaBank


7.	--Create a stored procedure that accepts AccountId as a parameter and returns customer’s full name. [Advanced]


create proc spFullNameFromAccountId2        
            @AccountID int,               
			                               
			@Fullname nvarchar(100) output 
as
begin
  if (@AccountID in (select AccountID from Customer_Account))
    begin
	  			Select @FullName= c.CustomerFirstName+' '+c.CustomerMiddleInitial+' '+c.CustomerLastName
				from Customer c
				join Customer_Account ca
				on ca.CustomerID=c.CustomerID
				where ca.AccountID=@AccountID;
               set @Fullname=replace (@FullName,'   ',' ')
   end
  else
   begin
    print 'This Account Id does not exists!'
   end
end

Declare @FullName nvarchar(100)
exec spFullNameFromAccountId2 5, @FullName out
Print ' Full name is '+@FullName

go

8.---	Create a stored procedure that returns error logs inserted in the last 24 hours. [Advanced]

create proc spError
as
--begin  
select * from LoginErrorLog
where ErrorTime >=  
DATEADD(Hour,-24,GETDATE());
--end 
select * from LoginErrorLog
insert into LoginErrorLog values ('6','2020-12-27 12:02','auto')
exec spError

go

9.	---Create a stored procedure that takes a deposit as a parameter and updates 
-----CurrentBalance value for that particular account. [Advanced]


create procedure spdeposit @accountID INT,

@deposit money 
As
select AccountID,currentbalance
from account where 
accountID=@accountID
update account 
set currentbalance=currentbalance+@deposit 
where accountID=@accountID;
select AccountID,currentbalance 
from account where 
accountID=@accountID


 
exec spdeposit 1,250

select *  from account





go


10.	----Create a stored procedure that takes a withdrawal amount as a parameter and updates 

create  proc updatecbalancewithdraw 

@accountID int,@withdraw money
as
update Account 
set currentbalance =currentbalance-@withdraw
where accountID= @accountID;
exec updatecbalancewithdraw 4,100

select * from Account

-----------------------------Thank you very much Mr Hamid ------------------- 
---------------------------Wish you Happy,Blessed, And safe holidays------------------
---------------------------Happy New Year ---------------------------