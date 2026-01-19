## Dataset: Credit Card Approval Prediction

There are two tables that can be merged using the `ID` column.

### application_record.csv

| Feature name | Explanation | Remarks |
|-------------|------------|---------|
| ID | Client number | Primary key |
| CODE_GENDER | Gender | |
| FLAG_OWN_CAR | Owns a car | |
| FLAG_OWN_REALTY | Owns a property | |
| CNT_CHILDREN | Number of children | |
| AMT_INCOME_TOTAL | Annual income | |
| NAME_INCOME_TYPE | Income category | |
| NAME_EDUCATION_TYPE | Education level | |
| NAME_FAMILY_STATUS | Marital status | |
| NAME_HOUSING_TYPE | Housing type | |
| DAYS_BIRTH | Birthday | Counted backwards from current day (0); -1 means yesterday |
| DAYS_EMPLOYED | Employment start date | Counted backwards from current day (0); positive value means currently unemployed |
| FLAG_MOBIL | Has a mobile phone | |
| FLAG_WORK_PHONE | Has a work phone | |
| FLAG_PHONE | Has a phone | |
| FLAG_EMAIL | Has an email | |
| OCCUPATION_TYPE | Occupation | |
| CNT_FAM_MEMBERS | Family size | |

### credit_record.csv

| Feature name | Explanation | Remarks |
|-------------|------------|---------|
| ID | Client number | Primary key |
| MONTHS_BALANCE | Record month | 0 = current month, -1 = previous month, and so on |
| STATUS | Credit status | 0: 1–29 days past due<br>1: 30–59 days past due<br>2: 60–89 days overdue<br>3: 90–119 days overdue<br>4: 120–149 days overdue<br>5: Overdue >150 days / bad debts<br>C: Paid off that month<br>X: No loan for the month |

## 🔗 Additional Information

- Related data: Credit Card Fraud Detection

- Related competition: Home Credit Default Risk