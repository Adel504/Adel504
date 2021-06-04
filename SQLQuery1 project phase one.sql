use db_CanadaBank
 select * from UserLogins
 select * from UserSecurityQuestions
 select * from AccountType
 select * from SavingsInterestRates
 insert into SavingsInterestRates values('1',0.04,'Basic')
 insert into SavingsInterestRates values('2',0.03,'premium')
 insert into SavingsInterestRates values('3',0.02,'low')
 insert into SavingsInterestRates values('4',0.01,'promotional')
 insert into SavingsInterestRates values('5',0.06,'saving')
 select *from AccountType
 insert into AccountType values ('3','money market account')
 insert into AccountType values ('4','Retirement')
 insert into AccountType values ('5','certificate of deposit')
 select * from Customer_Account
 select * from Account
  select *from AccountType
 insert into AccountType values ('2','certificate of deposit')
 select * from AccountStatusType
 select * from Account
 select * from AccountType
 select * from FailedTransactionErrorType
 select * from LoginErrorLog
 insert into  Customer_Account values(1,  1);
 insert into account values('1',11000,'2','3','4')
 insert into account values('9',21000,'1','2','3')
 insert into account values('3',45000,'3','1','2')
 insert into account values('4',6350,'4','4','1')
 insert into account values('5',7892,'5','5','5')
 insert into account values('4',6350,'11','12','13')
 insert into account values('5',7892,'14','15','16')
 use db_CanadaBank
 select * from AccountType
 select * from Customer_Account
 select * from account
  select * from AccountStatusType
 use db_CanadaBank
 select * from Login_Account
 select * from UserLogins
 select * from account
 select * from UserSecurityAnswer
 select * from UserSecurityQuestions
 insert into Login_Account values ('2','3')
 insert into Login_Account values ('3','4')
 insert into Login_Account values ('4','5')
 insert into Login_Account values ('5','9')
 insert into Login_Account values ('1','1')
select * from UserLogins
select * from UserSecurityQuestions
select * from UserSecurityAnswer
insert into UserSecurityAnswer values ('1','meshmesh','1')

insert into UserSecurityAnswer values ('2','Titanic','2')

insert into UserSecurityAnswer values ('3','Canada','3')

insert into UserSecurityAnswer values ('4','Mazda','4')


update customer 
set Province 
where=state
insert into UserSecurityAnswer values ('5','Westminister','5')
select * from Customer_Account
select * from Customer
replace ([state])'state' as 'province')
select * from account 
select* from Customer
select * from UserLogins
select * from Account
sp_rename 'customer.zipcode','postalcode'
sp_rename 'customer.SSN','SIN'
insert into Customer values ('5','9','102 kingstreet','10 Dundas street w','Lucy','W','Mcarthy','Toronto','ON','M9R2W4','mno@gmail.com','4169107079','6475211487','4166476698','1','5')

insert into Customer values ('2','3','40steele st','ali','S','Daei','Edmonton','AB','J7L5O9','def@gmail.com','4166055589','6471400487','4169485698','4','1')
insert into Customer values ('2','3','31 kepling ','john','M','smith','Toronto','ON','M9M2W4','abc@gmail.com','4169447589','6475696487','4169665698','7','2')
insert into Customer values ('3','4','10 victoria park ave','john','M','smith','Toronto','ON','M9M2W4','abc@gmail.com','4169117589','6475896487','4169785698','8','3')
insert into Customer values ('4','5','81 Bradford','john','M','smith','Toronto','ON','M9M2W4','abc@gmail.com','4169866689','6471296487','4169845698','9','4')
insert into Customer values ('5','9','102 Qween st','john','M','smith','Toronto','ON','M9M2W4','abc@gmail.com','4166367589','6476996487','41694575698','10','5')

use db_CanadaBank
select * from LoginErrorLog
select * from Customer_Account
select * from Account
select * from Customer
select * from OverDraftLog
select * from Account
insert into OverDraftLog values ('1','2020-12-21','15000','note')
insert into OverDraftLog values ('3','2020-12-22','4587','Roots')
insert into OverDraftLog values ('4','2020-12-23','8596','auto')
insert into OverDraftLog values ('5','2020-12-24','8000','Row')
insert into OverDraftLog values ('9','2020-12-25','74000','order')

select * from FailedTransactionLog
select * from FailedTransactionErrorType
insert into FailedTransactionLog values ('1','1','2020-12-10','xmldocument')
insert into FailedTransactionLog values ('2','2','2020-12-11','xmlhandle')
insert into FailedTransactionLog values ('3','3','2020-12-12','xmlname')
insert into FailedTransactionLog values ('4','4','2020-12-13','xmlgroup')
insert into  TransactionType values  ('1','Credit','payment',50 )
insert into  TransactionType values  ('5','secure','Direct deposit',30 )
insert into  TransactionType values  ('5','','charge ',15 )
insert into  TransactionType values  ('4','Credit','payment',50 )
insert into  TransactionLog values  ('5','2020-12-05 16:14:51','4',900,300,'9','5','5','5')

select * from TransactionLog
select * from Account
select * from TransactionType
select * from Customer
select * from Employee
select * from UserLogins
select * from Customer_Account

insert into Customer_Account values ('1','1')
insert into Customer_Account values ('2','3')
insert into Customer_Account values ('3','4')
insert into Customer_Account values ('4','5')
insert into Customer_Account values ('5','9')
use db_CanadaBank
