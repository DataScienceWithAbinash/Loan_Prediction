# Loan_Prediction
 A machine learning model to predict the loan approval probabilty.
 
 # Why it required?
 The two most critical questions in the lending industry are: 1) How risky is the borrower? 2) Given the borrower’s risk, should we lend him/her? The answer to the first question determines the interest rate the borrower would have. Interest rate measures among other things (such as time value of money) the riskness of the borrower, i.e. the riskier the borrower, the higher the interest rate. With interest rate in mind, we can then determine if the borrower is eligible for the loan.
 
 Investors (lenders) provide loans to borrowers in exchange for the promise of repayment with interest. That means the lender only makes profit (interest) if the borrower pays off the loan. However, if he/she doesn’t repay the loan, then the lender loses money.
 

### Import the required libraries
    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, f1_score,make_scorer
    from sklearn import tree
    import matplotlib.pyplot as plt
    import seaborn as sns
### Import the data
    Loan = pd.read_csv('Loan.csv')
    Loan.head()
### Check the description to understand something more about the data
    Loan.shape
    Loan.describe()
### Check missing values
    Loan.isnull().sum()
### Fill all the missing value
    Loan['Gender'].value_counts()
    Loan.Gender = Loan.Gender.fillna('Male')
    Loan['Married'].value_counts()
    Loan.Married = Loan.Married.fillna('Yes')
    Loan['Dependents'].value_counts()
    Loan.Dependents = Loan.Dependents.fillna('0')
    Loan['Self_Employed'].value_counts()
    Loan.Self_Employed = Loan.Self_Employed.fillna('No')
    Loan['LoanAmount'].value_counts()
    Loan.LoanAmount = Loan.LoanAmount.fillna(Loan.LoanAmount.mean())
    Loan['Loan_Amount_Term'].value_counts()
    Loan.Loan_Amount_Term=Loan.Loan_Amount_Term.fillna('360')
    Loan['Credit_History'].value_counts()
    Loan.Credit_History = Loan.Credit_History.fillna('1.0')
### Drop the Loan_Id column which is not important
    Loan = Loan.drop(['Loan_ID'],axis = 1)
    Loan.head()
### Replacing Values into numerical format
    Loan.Gender = Loan.Gender.replace({'Male':0,'Female':1})
    Loan.Married=Loan.Married.replace({'Yes':1,'No':0})
    Loan.Education = Loan.Education.replace({'Graduate':0,'Not Graduate':1})
    Loan.Self_Employed = Loan.Self_Employed.replace({'No':0,'Yes':1})
    Loan.Property_Area = Loan.Property_Area.replace({'Semiurban':0,'Urban':1,'Rural':2})
    Loan.Loan_status = Loan.Loan_status.replace({'N':0,'Y':1})
    Loan.Dependents = Loan.Dependents.replace({'3+':3})
    Loan.Credit_History = Loan.Credit_History.replace({'1.0':1})
### Separate the features and labels
    X=Loan.iloc[:,:-1].values
    y=Loan.iloc[:,11].values
### Split the data into training and testing
    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=6)
    print(f"Row in training set:{len(X_train)}\nRow in test set:{len(X_test)} ")
### Apply the the DecisionTreeClassifier 
    Loan_model = DecisionTreeClassifier()
    Loan_model.fit(X_train,y_train)
### Predict the model using test data  
