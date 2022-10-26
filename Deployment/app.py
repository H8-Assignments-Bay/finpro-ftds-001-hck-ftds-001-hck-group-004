import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

model = pickle.load(open('finpro_preproc_model.pkl', 'rb'))

st.image('soter.png', width=350)
st.title('Project SOTER')
st.header('Quick ID & Background Checking for Credit Risk Analysis')

st.header('Prospective Debtor Information')
st.text('Verification Status')
verification_status = st.selectbox('Select verification status', ['Source Verified', 'Verified', 'Not Verified'])
st.text('Grade')
grade = st.selectbox('Pick one', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
st.text('Sub Grade')
sub_grade = st.selectbox('Pick one', ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5'])
st.text('Loan Purpose')
#purpose = st.selectbox('Pick one', ['Debt Consolidation', 'Credit Card', 'Home Improvement', 'Moving', 'House', 'Small Business', 'Major Purchase', 'Medical', 'Car', 'Vacation', 'Wedding', 'Renewable Energy', 'Others'])
purpose = st.selectbox('Pick one', ['debt_consolidation', 'credit_card', 'home_improvement', 'moving', 'house', 'small_business', 'major_purchase', 'medical', 'car', 'vacation', 'wedding', 'renewable_energy', 'other'])
st.text('Employment Length in Years')
employment_length = st.selectbox('Pick one', ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
st.text('Home Ownership')
#home_ownership = st.selectbox('Pick one', ['Mortgage', 'Rent', 'Own', 'None'])
home_ownership = st.selectbox('Pick one', ['MORTGAGE', 'RENT', 'OWN', 'NONE', 'OTHER'])
st.text('Loan Initial Listing Status')
#initial_list_status = st.selectbox('Pick one', ['Whole Loan', 'Fractional Loan'])
initial_list_status = st.selectbox('w = Whole Loan | f = Fractional Loan', ['w', 'f'])

st.header("Debtor's Account Information")
st.text('Total Open Accounts')
open_account = st.number_input('Input number of open accounts')
st.text('Total Derogatory Public Records')
public_rec = st.number_input('Input number of records')
st.text('Total Accounts')
total_accounts = st.number_input('Input number of accounts')
st.text('Total Current Balance (All Accounts)')
total_cur_bal = st.number_input('Input total current balance on all accounts inn USD')
st.text('Total Credit Limit')
total_cred_lim = st.number_input('Input total credit limit in USD')
st.text('Total Payment')
total_payment = st.number_input('Input total payment in USD')
st.text('Revolving Line Utilization Rate')
revolving_util = st.number_input('Input rate')
st.text('Last Payment Amount')
last_payment_amnt = st.number_input('Input last payment amount in USD')
st.text('Total Revolving Balance')
revolving_bal = st.number_input('Input revolving balance')
st.text('Remaining Outstanding Principal Amount')
out_principal = st.number_input('Input amount in USD')
st.text('Gross Recovery')
recoveries = st.number_input('Input gross recovery')
st.text('Recovery Collection Fee')
collection_recov_fee = st.number_input('Input fee in USD')
st.text('Total Late Fees')
total_rec_late_fee = st.number_input('Input late fee in USD')
st.text('Principal Received to Date')
total_received_prncp = st.number_input('Input principal in USD')
st.text('Interest Received to Date')
total_received_int = st.number_input('Input interest rate')
st.text('Payments received to date for portion of total amount funded by investors')
total_payment_inv = st.number_input('Input payment in USD')
st.text('Remaining Outstanding Principals of Total Funded by Investors')
out_prncp_inv = st.number_input('Input outstanding principal in USD')

st.header('Loan Information')
st.text('Interest Rate')
interest_rate = st.number_input('Insert interest rate')
st.text('Loan Payment Term')
term = st.selectbox('Select the payment term', [36, 60])
st.text('Number of Inquiries (Last 6 Months)')
inquiry_last_6mths = st.number_input('Insert number of inquiry')
st.text('Installment Amount (per Month)')
installment = st.number_input('In USD')

if st.button('Submit'):
    num_cols = ['tot_cur_bal','total_rev_hi_lim','revol_util','inq_last_6mths','open_acc','pub_rec','total_acc','collection_recovery_fee','last_pymnt_amnt','recoveries','term','int_rate','installment',
                'total_rec_late_fee', 'revol_bal','out_prncp','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','out_prncp_inv']
    cat_col_ord = ['emp_length','verification_status','grade','sub_grade']
    cat_col_nom = ['home_ownership','purpose','initial_list_status']

    num_df = pd.DataFrame([[total_cur_bal, total_cred_lim, revolving_util, inquiry_last_6mths, open_account, public_rec, total_accounts, collection_recov_fee, last_payment_amnt, recoveries, term,
                            interest_rate, installment, total_rec_late_fee, revolving_bal, out_principal, total_payment, total_payment_inv, total_received_prncp, total_received_int, out_prncp_inv]],columns=num_cols)
    ordcat_df = pd.DataFrame([[employment_length, verification_status, grade, sub_grade]],columns=cat_col_ord)
    nomcat_df = pd.DataFrame([[home_ownership, purpose, initial_list_status]],columns=cat_col_nom)

    X = pd.concat([num_df, ordcat_df, nomcat_df],axis=1)
                       
    pred = model.predict(X)

    if pred == 0:
        st.image('https://findmywayhome.com/wp-content/uploads/2016/03/Get-an-Underwriting-Approval-Before-Buying-a-Home.jpg', width=200)
        st.text("Loan Accepted")
    elif pred == 1:
        st.image('https://thumbs.dreamstime.com/b/loan-rejected-rubber-stamp-over-white-background-86663998.jpg', width=200)
        st.text("Loan Denied")     
    else:
        st.text("Please Input Data.")
