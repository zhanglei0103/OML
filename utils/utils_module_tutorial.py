import sys
import os
import streamlit as st
import sklearn
from PIL import Image


def module_1_tutorial(module_1=None, image_folder_path=None):
    st.subheader(module_1)
    context_module1 = "The module's function is to enable users to upload datasets in formats of <span style='font-weight:bold'>csv, xlsx, or txt</span>. Upon successful upload, the user will receive the message <span style='font-weight:bold'>'File loaded successfully!'</span>. If the user clicks on the <span style='font-weight:bold'>'data preview' button</span>, they can preview the uploaded dataset on the left section of the home page. The row represents sample and the column represents feature or target variable. "
    revise_context_module1 = f'<p style="font-size: 18px;">{context_module1}</p>'
    st.markdown(revise_context_module1, unsafe_allow_html=True)
    if image_folder_path is not None:
        image_step_1 = Image.open(f"{image_folder_path}/step_1.png")
        st.image(image_step_1, use_column_width=True)



def module_2_tutorial(module_2=None, image_folder_path=None):
    st.subheader(module_2)
    context_module2 = "This module requires users to select the <span style='font-weight:bold'>sample identifier, target variable, and all independent variables</span> for the uploaded dataset. It's important to note that the platform does not provide any missing value imputation method, as there are many available methods and software for filling missing values in existing omics data. Additionally, any uploaded data features that contain missing values will be deleted during downstream analysis."
    revise_context_module2 = f'<p style="font-size: 18px;">{context_module2}</p>'
    st.markdown(revise_context_module2, unsafe_allow_html=True)
    if image_folder_path is not None:
        image_step_2 = Image.open(f"{image_folder_path}/step_2.png")
        st.image(image_step_2, width=600)


def module_3_tutorial(module_3=None, image_folder_path=None):
    st.subheader(module_3)
    context_module3 = "This module functions to normalize the dataset consisting of the user's selected target variable and all independent variables. The module provides various normalization methods, enabling users to select any one of them according to their needs. After normalizing the data, users can use the <span style='font-weight:bold'>'Normalized data preview' button</span> on the main page to preview and download the normalized dataset."
    revise_context_module3 = f'<p style="font-size: 18px;">{context_module3}</p>'
    st.markdown(revise_context_module3, unsafe_allow_html=True)
    if image_folder_path is not None:
        image_step_3 = Image.open(f"{image_folder_path}/step_3.png")
        st.image(image_step_3, use_column_width=True)

def module_4_tutorial(module_4=None, image_folder_path=None):
    st.subheader(module_4)
    context_module4 = "This is a module for <span style='font-weight:bold'>feature pre-selection and is optional</span>. Its specific function is to model <span style='font-weight:bold'>all independent variables</span> set by the user using two classic algorithms and sort them according to the importance values of the features in the model. If the user is unsure which features to select for modeling, it is recommended to execute this module and choose the subset of features with the highest sorting values for downstream processing."
    revise_context_module4 = f'<p style="font-size: 18px;">{context_module4}</p>'
    st.markdown(revise_context_module4, unsafe_allow_html=True)
    if image_folder_path is not None:
        image_step_4 = Image.open(f"{image_folder_path}/step_4.png")
        st.image(image_step_4, use_column_width=True)

def module_5_tutorial(module_5=None, image_folder_path=None):
    st.subheader(module_5)
    context_module5 = "This module allows users to select certain features, which are then used to construct downstream models."
    revise_context_module5 = f'<p style="font-size: 18px;">{context_module5}</p>'
    st.markdown(revise_context_module5, unsafe_allow_html=True)
    if image_folder_path is not None:
        image_step_5 = Image.open(f"{image_folder_path}/step_5.png")
        st.image(image_step_5, use_column_width=600)

def module_6_tutorial(module_6=None, image_folder_path=None):
    st.subheader(module_6)
    context_module6 = "This module's function is to divide the dataset into training and testing sets. It provides users with the option to choose between <span style='font-weight:bold'>Cross-Validation and Percentage Split Methods </span>based on their data attributes."
    revise_context_module6 = f'<p style="font-size: 18px;">{context_module6}</p>'
    st.markdown(revise_context_module6, unsafe_allow_html=True)
    if image_folder_path is not None:
        image_step_6 = Image.open(f"{image_folder_path}/step_6.png")
        st.image(image_step_6, use_column_width=600)

def module_7_tutorial(module_7=None, image_folder_path=None):
    st.subheader(module_7)
    context_module7 = f"The function of this module is to perform algorithmic modeling using the scikit-learn (v{sklearn.__version__}) package, which incorporates multiple algorithms. Users can select any of these algorithms to model selected datasets. This module is dependent on parameter settings from other modules and can only be properly executed when these parameters are reasonably configured. Each algorithm's default parameters correspond to those of scikitlearn's related algorithms. Users can modify these parameters themselves and choose the best hyperparameter combination based on the given model performance. Upon completion of modeling, a detailed document on model performance is generated, allowing users to download all graphs and tables. Alternatively, users can choose to use the browser's print function to retain all results generated during the process."
    revise_context_module7 = f'<p style="font-size: 18px;">{context_module7}</p>'
    st.markdown(revise_context_module7, unsafe_allow_html=True)
    if image_folder_path is not None:
        image_step_7 = Image.open(f"{image_folder_path}/step_7.png")
        st.image(image_step_7, use_column_width=300)


def get_tutorial_page(module_1, module_2, module_3, module_4, module_5,
                      module_6, module_7, image_folder_path):
    st.header('Process Flow Chart')
    context1 = 'OML is composed of seven analysis modules, each of which is easy to use and master. The fourth module is option, but others are required to set parameters. We recommend users first read the instructions of each module to quickly understand its functionality. The OML platform provides an assortment of example data for both classification and regression models, each with default parameter settings across all seven modules. These sample data sets are utilized to demonstrate how to configure the relevant parameters for each module and highlight relevant considerations.'
    revise_context1 = f'<p style="font-size: 18px;">{context1}</p>'
    st.markdown(revise_context1, unsafe_allow_html=True)
    image_flowchart = Image.open(f"{image_folder_path}/flow chart.png")
    st.image(image_flowchart, use_column_width=True)
    module_1_tutorial(module_1, image_folder_path)
    module_2_tutorial(module_2, image_folder_path)
    module_3_tutorial(module_3, image_folder_path)
    module_4_tutorial(module_4, image_folder_path)
    module_5_tutorial(module_5, image_folder_path)
    module_6_tutorial(module_6, image_folder_path)
    module_7_tutorial(module_7, image_folder_path)


def get_disclaimer(disclaimer):
    if disclaimer:
        st.write("***")
        st.header('Disclaimer')
        context_disclaimer = "The OML (Omics Machine Learning) online analysis platform is designed to assist users in modeling and selecting biomarkers for omics data, based on the best practices and research findings of scholars. It provides a simple modeling process that can be quickly learned by users with any programming and machine learning background. However, we cannot guarantee the complete accuracy of the analysis results or their suitability for specific use cases due to various factors affecting model performance, such as study design, dataset attributes, feature selection, algorithm selection, and algorithm parameter settings. Therefore, we cannot be held responsible for any negative consequences resulting from the use of our platform or its contents."
        revise_context_disclaimer = f'<p style="font-size: 18px;">{context_disclaimer}</p>'
        st.markdown(revise_context_disclaimer, unsafe_allow_html=True)
        st.write("***")
