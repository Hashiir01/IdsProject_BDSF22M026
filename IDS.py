import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load():
    data=pd.read_excel('C:/Users/HP/OneDrive/Desktop/IDS_PROJECT/DATA SET  EMPLOYEE  RETENTION.xlsx')
    data=data.drop('Asupervisor treated you  with consideration when giving you your performance appraisal results',axis=1)
    data=data.drop('Employee number',axis=1)
    return data

data=load()
# print(data.shape)

#*****************Rename Columns*************************
rename= {
        'I am Enjoying my job': 'Job Enjoyment',
        'Potential promotion opportunities ': 'Promotion Opportunities',
        'Good relationship with supervisor /managers': 'Relation with supervisor',
        'supervisor rated you on how well you did your job, not on his/her personal opinion of you': 'Fair rating',
        'Sufficient time is allocated for product and solution training': 'Training time',
        'The supervisor that evaluated you showed concern for your rights as an employee': 'Supervisor Concern',
        'Overall, how hard did the supervisor who rated your performance try to be fair to you': 'Fairness',
        'Overall, how fairly were you treated by the supervisor who rated your performance': 'Fair Treatment',
        'It would be easy to find a job in another department': 'Job Mobility',
        'My supervisor rewards a good idea by implementing it and giving the responsible employee( s) credit': 'Reward of Ideas',
        'I am Rewarded whenever I do a good job': 'Reward for good job',
        'My chances for being promoted are good ': 'Promotion chances'
    }
data.rename(columns=rename, inplace=True)

st.set_page_config(
    page_title="Employee Retention Analysis", 
    page_icon="📊", 
)

#**************Header**********************
header_html = """
    <style>
        .header-container {
            width: 100%;
            background-color: #0E1117;
            padding: 5px; 
            margin-bottom: 20px;
            
        }
        .header-text {
            font-family: Arial, sans-serif;
            color: white;
            text-align: center;
            font-size: 30px; 
        }
    </style>
    <div class="header-container">
        <h1 class="header-text">IDS PROJECT</h1>
    </div>
"""
st.markdown(header_html, unsafe_allow_html=True)


#**************Footer**********************

footer_style = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0E1117;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #ffffff;
        z-index: 1000; /* Ensures it stays above other elements */
    }
    .main-content {
        padding-bottom: 50px; /* Adds space for the footer */
    }
    </style>
    <div class="footer">
        © 2025 Muhammad Hashiir | All Rights Reserved
    </div>
"""

st.markdown(footer_style, unsafe_allow_html=True)
st.markdown('<div class="main-content"></div>', unsafe_allow_html=True)




#************* Title *******************************
st.title('Employee Retention Dataset Analysis')

#************* SideBar *****************************
st.sidebar.title('Analysis')
page = st.sidebar.radio("Select Analysis Page:", [
    "Overview","Demographics", "Income Analysis", "Retention Insights", "Job Satisfaction", "ML Model"
])

if page=="Overview":
    st.header("Dataset Overview")
    st.write("Shape Of Data")
    st.write("**Shape of the dataset:**", data.shape)
    st.write('## The Data is Based on Ratings of Employees')
    st.write('**1 means The Lowest Rating and 5 means the Highest Rating except Age and Experience**')
    st.dataframe(data.head(10))
    
elif page=='Demographics':
    st.header("Demographics Analysis")
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='Age', kde=True, color='darkblue', bins=15, ax=ax)
    st.pyplot(fig)
    st.write('**Fig 1.1  Shows The Count Of Employees With Respect to Age.**')

    st.subheader("Gender Distribution")
    gender_mapping = {1: 'Male', 2: 'Female'}
    data['Gender'] = data['Gender'].map(gender_mapping)
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='Gender', palette='Set2', ax=ax)
    st.pyplot(fig)
    st.write('**Fig 1.2  Shows The Count Male and Female Employees.**')

    st.subheader("Marital Status Distribution")
    martial = {1: 'Married', 2: 'Unmarried'}
    data['Marital status '] = data['Marital status '].map(martial)
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='Marital status ', palette='Set1', ax=ax)
    st.pyplot(fig)
    st.write('**Fig 1.3  Shows The Count Of Married and Unmarried Employees.**')

    st.subheader('Designation by Martial Status')
    fig = plt.figure()
    sns.violinplot(x="Marital status ", y="Designation", data=data, palette="muted")
    plt.title("Years in Role by Marital Status")
    plt.xlabel("Marital Status")
    plt.ylabel("Designation")
    st.pyplot(fig)
    st.write('**Fif 1.4 Shows the Designation of Employees Based on Marital Status.**')

elif page == "Income Analysis":
    st.header("Income Analysis")
    st.subheader("Monthly Income by Designation")
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x='Designation', y='Income', palette='viridis', ax=ax)
    st.pyplot(fig)
    st.write('**Fig 1.5  Shows The Monthly Income of Employees Based on their Designation**')
    
    st.subheader("Job Enjoyment vs Monthly Income")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='Job Enjoyment', y='Income', hue='Designation', palette='deep', ax=ax)
    st.pyplot(fig)
    st.write('**Fig 1.6  Shows The Relationship Between Job Enjoyment and Monthly Income of Employees**')

    st.subheader("Distribution of Monthly Income")
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='Income', kde=True, color='green', bins=50, ax=ax)
    st.pyplot(fig)
    st.write("**Fig 1.7  Shows The Distribution of Monthly Income among the Employees**")


elif page == "Retention Insights":
    st.header("Retention Insights")
    st.subheader("Designation vs Promotion Oppertunities")
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x='Designation', y='Promotion Opportunities', palette='bright', ax=ax)
    st.pyplot(fig)
    st.write('**Fig 1.8  Shows The Promotion Opportunities of Employees Based on Designation.**')

    st.subheader("Age Distribution by Experience")
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x='Experience', y='Age', palette='cool', ax=ax)
    st.pyplot(fig)
    st.write('**Fig 1.9  Shows The Experience of Employees according to Age.**')

elif page == "Job Satisfaction":
    st.header('Job Satisfaction')
    st.subheader("Job Satisfaction by Job Mobility")
    fig, ax = plt.subplots()
    sns.barplot(data=data, x='Job Mobility', y='Job Enjoyment', ci=None, palette='cool', ax=ax)
    st.pyplot(fig)
    st.write('**Fig 2.0 Shows The Job Enjoyment of Employees Based on Job Mobility.**')

    st.subheader("Job Satisfaction by Reward")
    fig, ax = plt.subplots()
    sns.barplot(data=data, x='Reward for good job', y='Job Enjoyment', ci=None, palette='deep', ax=ax)
    st.pyplot(fig)
    st.write('**Fig 2.1 Shows The Job Enjoyment of Employees Based on Reward for Good Job**')

elif page=='ML Model':
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler

    features = ['Age', 'Experience', 'Income', 'Designation', 'Fairness', 'Job Mobility', 'Reward of Ideas']
    X = data[features]
    y = data['Job Enjoyment']  

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    st.header("Retention Prediction")
    st.write("Fill out the following details to predict if an employee is likely to be retained.")

    age = st.slider("Age", int(data['Age'].min()), int(data['Age'].max()), 30)
    experience = st.slider("Experience", int(data['Experience'].min()), int(data['Experience'].max()), 5)
    income = st.slider("Income", int(data['Income'].min()), int(data['Income'].max()), 3)
    designation = st.selectbox("Designation", sorted(data['Designation'].unique()))
    fairness = st.slider("Supervisor Fairness", 1, 5, 3)
    job_mobility = st.slider("Job Mobility", 1, 5, 3)
    reward_ideas = st.slider("Reward for Ideas", 1, 5, 3)

    user_data = pd.DataFrame({
        'Age': [age],
        'Experience': [experience],
        'Income': [income],
        'Designation': [designation],
        'Fairness': [fairness],
        'Job Mobility': [job_mobility],
        'Reward of Ideas': [reward_ideas]
    })
    user_data_scaled = scaler.transform(user_data)

    if st.button("Predict"):
        prediction = model.predict(user_data_scaled)
        ret = "Retention" 
        if prediction [0] == 0:
            ret=='No Retention'
        st.write(f"**Prediction:** {ret}")
