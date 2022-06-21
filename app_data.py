import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time
from helper import changefont

def app():
    # changefont()
    if 'keep_alive' not in st.session_state:
        st.session_state.keep_alive = False
    st.markdown('')
    title = '<p style = "font-family: PacFont; text-align:left; margin-bottom:0; color: Black; font-size: 30px;">WELCOME To Ai/O </p>'
    st.markdown(title, unsafe_allow_html=True)    
    st.text("")
    st.text("")
    st.write("Perform Linear Regression 🤖")
    st.write("First Select an appropriate file to analyze.")
    uploaded_file = st.file_uploader("Choose a file.")

    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Here is a view of your dataset!")
        st.write(df)
        st.text("")
        st.write("Now proceed to select the variable that you wish to predict (also known as response variable or Y value) for future values.")
        response = st.radio("Select the response variable for prediction: ", options = df.columns.tolist())
        st.text("")
        st.text("")
        st.write("Now proceed to select the variable that you wish to 'use' (also known as predictor variable or X value) to predict responses.")
        predictor = st.radio("Select the predictor variable for prediction: ", options = df.columns.tolist())
        fit, predict = st.columns((1,2))
        next_1 = fit.button("Perform Linear Regression")       
            
        if next_1:
            st.session_state.keep_alive = True
        if st.session_state.keep_alive == True:

            # Assign TV advertising as predictor variable 'x' and sales as response variable 'y'
            x = df[[predictor]]
            y = df[response]


            #Split the dataset in training and testing with 80% training set & set random state to 310 for reproducable results 
            x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=310)            

            # Use the sklearn function 'LinearRegression' to fit on the training set
            model = LinearRegression()   

            st.title("Your Linear Regression Model Training has begun.")
            st.write("")
            time.sleep(0.5)
            model.fit(x_train, y_train)
            # Now predict on the test set
            y_pred_test = model.predict(x_test)

            # Now compute the MSE with the predicted values and print it
            mse = mean_squared_error(y_test, y_pred_test)
            st.subheader('The test Mean Squared Error for your model is '+ str(np.round(mse,2)))
            st.text("")
            
            fig_out, ax_out = plt.subplots(figsize=(5,5))
            ax_out.set_title("Overview of your trained Model")
            ax_out.scatter(x_train, y_train, edgecolor = 'k',label='training set',color = '#B9D6D6')
            ax_out.scatter(x_test, y_test, edgecolor = 'k',label='test set',color = 'darkblue')
            ax_out.plot(x_test, y_pred_test, label='model',color = '#E44E47',lw=2)
            ax_out.set_xlabel(str(predictor))
            ax_out.set_ylabel(str(response))
            ax_out.legend()   

            
            st.pyplot(fig_out)

            cols = st.columns(2)
            number = cols[0].number_input("Go and ahead and enter your values for prediction!")
            inp = np.array(number).reshape(1,-1)
            
            st.subheader("Increase in "+str(response)+": "+str(int(model.predict(inp)[0])) + " units")                                                      

    






            
