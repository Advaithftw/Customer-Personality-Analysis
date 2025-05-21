import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import datetime
from datetime import date
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(layout="wide", page_title="Customer Personality Analysis")
st.title("Customer Personality Analysis")

st.sidebar.header("About Dataset")
st.sidebar.markdown("""
### Problem Statement
Customer Personality Analysis helps a business to better understand its customers and makes it easier to modify products according to the specific needs, behaviors and concerns of different types of customers.
""")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Customer_Segmentation_Dataset.csv")
        return df
    except:
        st.error("Please Upload the Dataset")
        st.stop()
        return None

def main():
    df = load_data()
    tabs = st.tabs(["Data Overview", "Data Preparation", "Univariate Analysis", "Bivariate Analysis", "Clustering"])
    
    with tabs[0]:
        st.header("Data Overview")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Shape:", df.shape)
        with col2:
            st.write("Missing Values:", df.isna().sum().sum())
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe().T)
    
    with tabs[1]:
        st.header("Data Preparation")
        
        st.subheader("Handling Missing Values")
        st.write("Filling missing Income values with median")
        
        df_clean = df.copy()
        df_clean['Income'] = df_clean['Income'].fillna(df_clean['Income'].median())
        
        st.subheader("Dropping Unnecessary Columns")
        df_clean = df_clean.drop(columns=["Z_CostContact", "Z_Revenue"], axis=1)
        
        st.subheader("Education Categories")
        df_clean['Education'] = df_clean['Education'].replace(['PhD','2n Cycle','Graduation', 'Master'],'Post Graduate')
        df_clean['Education'] = df_clean['Education'].replace(['Basic'], 'Under Graduate')
        
        st.subheader("Marital Status Categories")
        df_clean['Marital_Status'] = df_clean['Marital_Status'].replace(['Married', 'Together'],'Relationship')
        df_clean['Marital_Status'] = df_clean['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')
        
        st.subheader("Creating New Features")
        df_clean['Kids'] = df_clean['Kidhome'] + df_clean['Teenhome']
        df_clean['Expenses'] = df_clean['MntWines'] + df_clean['MntFruits'] + df_clean['MntMeatProducts'] + df_clean['MntFishProducts'] + df_clean['MntSweetProducts'] + df_clean['MntGoldProds']
        df_clean['TotalAcceptedCmp'] = df_clean['AcceptedCmp1'] + df_clean['AcceptedCmp2'] + df_clean['AcceptedCmp3'] + df_clean['AcceptedCmp4'] + df_clean['AcceptedCmp5']
        df_clean['NumTotalPurchases'] = df_clean['NumWebPurchases'] + df_clean['NumCatalogPurchases'] + df_clean['NumStorePurchases'] + df_clean['NumDealsPurchases']
        df_clean['Customer_Age'] = (pd.Timestamp('now').year) - df_clean['Year_Birth']
        
        st.subheader("Converting Customer Join Date")
        # Inspect unique values for debugging (optional, can be removed in production)
        st.write("Unique values in Dt_Customer:", df_clean['Dt_Customer'].unique())
        
        # Convert Dt_Customer to datetime, assuming format is DD-MM-YYYY
        df_clean["Dt_Customer"] = pd.to_datetime(df_clean["Dt_Customer"], format="%d-%m-%Y", errors='coerce')
        
        # Drop rows with NaT in Dt_Customer
        df_clean = df_clean.dropna(subset=['Dt_Customer'])
        
        # Calculate Customer_For (days since joining relative to max date)
        dates = df_clean["Dt_Customer"].dt.date
        d1 = max(dates)
        days = [(d1 - i).days for i in dates]
        df_clean["Customer_For"] = days
        
        col_del = ["Year_Birth", "ID", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", 
                  "AcceptedCmp5", "NumWebVisitsMonth", "NumWebPurchases", "NumCatalogPurchases", 
                  "NumStorePurchases", "NumDealsPurchases", "Kidhome", "Teenhome", "MntWines", 
                  "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds",
                  "Dt_Customer", "Recency", "Complain", "Response"]
        
        df_clean = df_clean.drop(columns=col_del, axis=1)
        
        st.subheader("Removing Outliers")
        df_clean = df_clean[df_clean['Customer_Age'] < 90]
        df_clean = df_clean[df_clean['Income'] < 300000]
        
        st.subheader("Final Processed Data")
        st.dataframe(df_clean.head())
        
        st.session_state['df_clean'] = df_clean
    
    with tabs[2]:
        if 'df_clean' not in st.session_state:
            st.error("Please visit Data Preparation tab first")
            return
            
        df_clean = st.session_state['df_clean']
        st.header("Univariate Analysis")
        
        feature = st.selectbox("Select Feature", 
                              ["Education", "Marital_Status", "Income", "Kids", 
                               "Expenses", "TotalAcceptedCmp", "NumTotalPurchases", 
                               "Customer_Age", "Customer_For"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if feature in ["Education", "Marital_Status", "Kids", "TotalAcceptedCmp", "NumTotalPurchases"]:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(x=feature, data=df_clean)
                plt.xticks(rotation=90)
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df_clean[feature], kde=True)
                st.pyplot(fig)
                
        with col2:
            if feature in ["Income", "Expenses", "Customer_Age", "Customer_For"]:
                fig, ax = plt.subplots(figsize=(10, 6))
                df_clean[feature].plot.box()
                st.pyplot(fig)
            else:
                st.write(df_clean[feature].value_counts())
    
        with tabs[3]:
            if 'df_clean' not in st.session_state:
                st.error("Please visit Data Preparation tab first")
                return
                
            df_clean = st.session_state['df_clean']
            st.header("Bivariate Analysis")
            
            x_feature = st.selectbox("Select X Feature", 
                                    ["Education", "Marital_Status", "Kids", 
                                     "TotalAcceptedCmp", "NumTotalPurchases", "Customer_Age"])
            
            y_feature = st.selectbox("Select Y Feature", 
                                    ["Expenses", "Income", "NumTotalPurchases"], index=0)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            if x_feature in ["Education", "Marital_Status", "Kids", "TotalAcceptedCmp"]:
                sns.barplot(x=x_feature, y=y_feature, data=df_clean)
            else:
                sns.scatterplot(x=x_feature, y=y_feature, data=df_clean)
            plt.title(f"{x_feature} vs {y_feature}")
            plt.xticks(rotation=90)
            st.pyplot(fig)
            
            st.subheader("Correlation Matrix")
            # Select only numeric columns for correlation
            numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
            plt.figure(figsize=(10, 8))
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='Greys', linewidths=1)
            st.pyplot(fig)
    
    with tabs[4]:
        if 'df_clean' not in st.session_state:
            st.error("Please visit Data Preparation tab first")
            return
            
        df_clean = st.session_state['df_clean']
        st.header("Clustering")
        
        st.subheader("Data Preprocessing for Clustering")
        
        X = df_clean.copy()
        
        label_encoder = preprocessing.LabelEncoder()
        X['Education'] = label_encoder.fit_transform(X['Education'])
        X['Marital_Status'] = label_encoder.fit_transform(X['Marital_Status'])
        
        scaler = StandardScaler()
        col_scale = ['Income', 'Kids', 'Expenses', 'TotalAcceptedCmp', 
                    'NumTotalPurchases', 'Customer_Age', 'Customer_For']
        X[col_scale] = scaler.fit_transform(X[col_scale])
        
        clustering_method = st.radio("Select Clustering Method", ["K-Means", "Agglomerative Clustering"])
        
        if clustering_method == "K-Means":
            st.subheader("K-Means Clustering")
            
            st.write("Finding Optimal Number of Clusters using Elbow Method")
            
            col1, col2 = st.columns(2)
            
            with col1:
                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                    kmeans.fit(X)
                    wcss.append(kmeans.inertia_)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(range(1, 11), wcss, 'bx-')
                plt.title('The Elbow Method')
                plt.xlabel('Number of clusters')
                plt.ylabel('WCSS')
                st.pyplot(fig)
            
            with col2:
                n_clusters = st.slider("Select Number of Clusters", 2, 10, 2)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
                pred = kmeans.predict(X)
                X['cluster_Kmeans'] = pred + 1
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(x=X["cluster_Kmeans"])
                plt.title("Distribution Of The Clusters")
                st.pyplot(fig)
            
            st.subheader("Visualization of Clusters")
            
            x_var = st.selectbox("Select X Variable", 
                              ["Expenses", "Income", "Kids", "Customer_Age", 
                               "Marital_Status", "NumTotalPurchases"], index=0)
            
            y_var = st.selectbox("Select Y Variable", 
                              ["Income", "Expenses", "Kids", "Customer_Age",
                               "Marital_Status", "NumTotalPurchases"], index=1)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(x=x_var, y=y_var, hue='cluster_Kmeans', data=X)
            plt.title(f"Clusters Visualization: {x_var} vs {y_var}")
            st.pyplot(fig)
            
        else:
            st.subheader("Agglomerative Clustering with PCA")
            
            pca = PCA(n_components=3)
            pca.fit(X)
            PCA_ds = pd.DataFrame(pca.transform(X), columns=(["col1", "col2", "col3"]))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("PCA Components Explained Variance Ratio:", pca.explained_variance_ratio_)
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(PCA_ds["col1"], PCA_ds["col2"], PCA_ds["col3"], c="maroon", marker="o")
                ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
                st.pyplot(fig)
            
            with col2:
                n_clusters = st.slider("Select Number of Clusters", 2, 10, 2)
                
                AC = AgglomerativeClustering(n_clusters=n_clusters)
                yhat_AC = AC.fit_predict(PCA_ds)
                PCA_ds["Clusters"] = yhat_AC
                X["Cluster_Agglo"] = yhat_AC + 1
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(x=X["Cluster_Agglo"])
                plt.title("Distribution Of The Clusters")
                st.pyplot(fig)
            
            st.subheader("3D Visualization of Clusters")
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(PCA_ds["col1"], PCA_ds["col2"], PCA_ds["col3"], 
                      s=40, c=PCA_ds["Clusters"], marker='o')
            ax.set_title("The Plot Of The Clusters")
            st.pyplot(fig)
            
            st.subheader("Clusters Visualization")
            
            x_var = st.selectbox("Select X Variable", 
                              ["Expenses", "Income", "Kids", "Customer_Age", 
                               "Marital_Status", "NumTotalPurchases"], index=0)
            
            y_var = st.selectbox("Select Y Variable", 
                              ["Income", "Expenses", "Kids", "Customer_Age",
                               "Marital_Status", "NumTotalPurchases"], index=1)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(X[x_var], X[y_var], hue=X['Cluster_Agglo'])
            plt.title(f"Clusters Visualization: {x_var} vs {y_var}")
            st.pyplot(fig)
        
        st.subheader("Cluster Insights")
        st.markdown("""
        ### Cluster 1:
        - People with less expenses
        - People who are married and parents of more than 3 kids
        - People with low income
        
        ### Cluster 2:
        - People with more expenses
        - People who are single or parents who have less than 3 kids
        - People with high income
        - Age is not the main criteria but it is observed to some extent that older people fall in this group
        
        So, the customers falling in cluster 2 tend to spend more. The company can target people in cluster 2 for the sale of their products.
        """)

if __name__ == "__main__":
    main()