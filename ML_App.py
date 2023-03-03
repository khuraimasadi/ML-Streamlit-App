from pyforest import*
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import streamlit as st
from PIL import Image
st.title('Total Machine Learning')
st.image(Image.open('download.png'), use_column_width=True)
def main():
	activties = ['EDA', 'Visualization', 'Model', 'About Us']
	option = st.sidebar.selectbox('Select Option', activties)
	if option == 'EDA':
		st.subheader('Exploratory Data Analysis')
		data = st.file_uploader('Upload dataset', type=['csv', 'xls', 'txt', 'json'])
		if data is not None:
			st.success('Uploaded successfully')
			data = pd.read_csv(data)
			df=data
			st.dataframe(data.head(10))
			if st.checkbox('Display Shape'):
				st.write(data.shape)
			if st.checkbox('Display Columns'):
				st.dataframe(pd.DataFrame(data.columns))
			if st.checkbox('Select multiple columns'):
				selected_columns=st.multiselect('Select',data.columns)
				df=data[selected_columns]
				st.dataframe(df)
			if st.checkbox('Display Summary'):
				st.dataframe(pd.DataFrame(df.describe().T))
			if st.checkbox('Display Null Values'):
				st.dataframe(pd.DataFrame(data.isna().sum()))
			if st.checkbox('Display Data Types'):
				st.dataframe(pd.DataFrame(data.dtypes))
			if st.checkbox('Display Correlation'):
				st.dataframe(pd.DataFrame(data.corr()))	
	elif option=='Visualization':
		st.subheader("Visualization")
		data = st.file_uploader('Upload dataset', type=['csv', 'xls', 'txt', 'json'])
		if data is not None:
			data=pd.read_csv(data)
			df=data
			st.dataframe(df)
			if st.checkbox('Select multiple columns'):
				cols=st.multiselect('Select',data.columns)
				df=data[cols]
				st.dataframe(df)
			if st.checkbox('Display heatmap'):
				fig=plt.figure()
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot(fig)	
			if st.checkbox('Display distplot'):
				for col in df.describe().columns:
					fig=plt.figure()
					st.write(sns.distplot(df[col]))
					st.pyplot(fig)
			if st.checkbox('Display countplot'):
				for col in df.describe().columns:
					fig=plt.figure()
					st.write(sns.countplot(df[col]))
					st.pyplot(fig)	
			if st.checkbox('Display boxplot'):
				for col in df.describe().columns:
					fig=plt.figure()
					st.write(sns.boxplot(df[col]))
					st.pyplot(fig)
			if st.checkbox('Display pairplot'):
				fig=plt.figure()
				st.write(sns.pairplot(data=df))	
				st.pyplot(fig)	
			if st.checkbox('Display piechart'):	
				fig=plt.figure()
				columns=st.selectbox('Select columns',data.columns)
				piechart=data[columns].value_counts().plot.pie(autopct='%1.1f%%')
				st.pyplot(fig)
	elif option=='Model':
		count=0;
		st.subheader("Model Building")
		data = st.file_uploader('Upload dataset', type=['csv', 'xls', 'txt', 'json'])
		if data is not None:
			data=pd.read_csv(data)
			df=data
			st.dataframe(data.head(10))
			if st.checkbox('Select multiple columns'):
				count=1
				cols=st.multiselect('Select your preferred columns',data.columns)
				df=data[cols]
				st.dataframe(df)
				if df is not None:
					x=df.iloc[:,:-1]
					y=df.iloc[:,-1]
			seed=st.sidebar.slider('Seed',1,20,1)
			name=st.sidebar.selectbox('Select Classifier',('SVM','KNN','LogisticRegression'))
			model=None
			if name=='KNN':
				k=st.sidebar.slider('Neighbors',1,50,5)
				model=KNeighborsClassifier(n_neighbors=k)
			elif name=='SVM':
				c=st.sidebar.slider('C',1,100,5)
				model=SVC(C=c)
			else:
				model=LogisticRegression()
			if count==1:
				x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=seed)
				model.fit(x_train,y_train)
				score=model.score(x_test,y_test)
				st.write('Classifier name: ',name)
				st.write('Model performance: {:2f}%'.format(score*100))	

	else:
		st.subheader('About Us')
		st.markdown('This is an interactive web page for machine learning models. You can check and compare performance of different models here. This is dataset is fetched from UCI machine learning repository and kaggle. You can present your work to your stakeholders in an interactive way using different datasets.')
		st.balloons()		
				

main()				