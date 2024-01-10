
import os
import sys

import copy
import pandas as pd
from pandas.api.types import is_object_dtype, is_integer_dtype
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
import pathlib
from bs4 import BeautifulSoup
import logging
import shutil

from sklearn.model_selection import train_test_split

from st_aggrid import AgGrid


from utils.utils_module_tutorial import *
from utils.utils_upload_check_data import LoadDatasetCheck, load_dataset
from utils.utils_feture_selection_from_model import feature_select_from_model_table
from utils.utils_ML_method import ml_model_set_paras
from utils.utils_ML_model_fit_predict import ml_model_run
from utils.utils_split_data import select_cross_validation
from utils.utils_feature_normalize import feature_trans
from utils.utils_plot import (ml_model_show_summary_table, ml_model_show_figure)
from utils.utils_other_helps import convert_df

image_folder_path = "OML_logo"
image_logo = Image.open(f"{image_folder_path}/OML_Logo.png")
st.set_page_config(page_title='OML', page_icon=image_logo, layout='wide')
st.sidebar.image(image_logo, use_column_width=True)
st.sidebar.markdown("<h1 style='text-align: center; color: black; font-weight: bold;'>version: 0.1.0</h1>", unsafe_allow_html=True)

st.title('Welcome to Omics Machine Learn')
context0 = f'<span style="font-weight:bold">OML(Omics Machine Learning)</span> online analysis platform is built on the Streamlit package (v{st.__version__}) and Python (v3.9+) programming environment. It allows users to upload data and set various parameters related to their desired machine learning algorithm. This enables the platform to automatically create a data model based on the uploaded data, which then outputs detailed performance evaluation results. OML is not require extensive programming skills or knowledge of various algorithm models to build machine learning models.'
original_context0 = f'<p style="font-size: 18px;">{context0}</p>'
st.markdown(original_context0, unsafe_allow_html=True)
st.write("***")


modules = {
    "module_1": "__1\. Upload Dataset (⭐)__",
    "module_2": '__2\. Select Target Variable and Independent Variable (⭐)__',
    "module_3": '__3\. Normalize Dataset (⭐)__',
    "module_4": '__4\. Feature Selection from Model (Optional)__',
    "module_5": '__5\. Select Important Features (⭐)__',
    "module_6": '__6\. Split Data into Training Set and Testing Set (⭐)__',
    "module_7": '__7\. Choose Machine Learning Method (⭐)__',
}


with st.sidebar.expander("__OML Tutorial__", expanded=True):
    show_flowchart = st.checkbox("OML Process Flow Chart", value="OML Process Flow Chart")
    show_disclaimer = st.checkbox("Disclaimer", value="Disclaimer")

if show_flowchart:
    get_tutorial_page(**modules, image_folder_path=image_folder_path)

if show_disclaimer:
    get_disclaimer(disclaimer=True)

with st.sidebar.expander(modules["module_1"], expanded=False):
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt', 'xlsx', 'xls'])

    example_file = st.selectbox('Example dataset',
                                ['None',
                                 'Classification: Immune Checkpoint Blockade',
                                 "Classification: Alzheimer's disease",
                                 'Classification: Pan-cancer',
                                 'Regression: Gestational Age'],
                                help="Use in-built example file to demo the app")
    if 'Alzheimer' in example_file:
        default_filename = './OML_example_dataset/Alzheimer.xlsx'
    elif 'Gestational' in example_file:
        default_filename = "./OML_example_dataset/Metabolic_GestationalAge.xlsx"
    elif 'Immune' in example_file:
        default_filename = "./OML_example_dataset/Immune_Checkpoint_Blockade.xlsx"
    elif 'Pan-cancer' in example_file:
        default_filename = "./OML_example_dataset/Olink_U-CAN_Pan-cancer_1463proteins_NPX.txt"
    else:
        default_filename = None

    if (uploaded_file is None) & (example_file == 'None'):
        df = pd.DataFrame()
        sample_identifier = None
        sample_identifier_option = ['None']
        targetvalue_column_option = ['None']
        features_list = []
        default_features = []
        text_feature_value = "Example:\nfeature1\nfeature2\n..."
    elif uploaded_file is not None:
        if example_file != 'None':
            st.warning(
                "Warning: You have the option to select a file from your local machine by clicking the 'Browse' button below. Additionally, you may choose to use the provided example file by selecting the 'Example dataset' option. Please note that selecting both options will result in the local file being used.")
        load_df = load_dataset(uploaded_file)
        df = copy.deepcopy(load_df)
        # the initial values for sample_identifier in select box
        sample_identifier_option = ['None'] + list(df.columns)
        # the initial values for target feature in select box
        targetvalue_column_option = ['None'] + list(df.columns)
        # # the initial values for features in select box
        features_list = list(df.columns)
        default_features = None
        text_feature_value = ""
    elif (uploaded_file is None) & ('Gestational' in default_filename):
        ldc = LoadDatasetCheck(default_filename, targetvalue_column=None).load_dataset(set_index_col=None,
                                                                                       dropna_row=False)
        load_df = ldc.df.copy(deep=True)
        df = ldc.df.copy(deep=True)
        sample_identifier_option = ['None', 'Subject ID'] + list(df.drop(columns='Subject ID').columns)
        targetvalue_column_option = ['Gestational age (GA)/weeks'] + list(df.drop(columns='Gestational age (GA)/weeks').columns)
        features_list = list(df.columns)
        default_features = ['PE(P-16:0e/0:0)', 'Progesterone', 'Dehydroisoandrosterone sulfate', 'THDOC', 'Estriol-16-Glucuronide']
        text_feature_value = '\n'.join(default_features)
    elif (uploaded_file is None) & ('Alzheimer' in default_filename):
        ldc = LoadDatasetCheck(default_filename, targetvalue_column=None).load_dataset(set_index_col=None,dropna_row=False)
        load_df = ldc.df.copy(deep=True)
        df = ldc.df.copy(deep=True)
        sample_identifier_option = ['None'] + list(df.columns)
        targetvalue_column_option = ['_primary biochemical AD classification'] + list(df.drop(columns='_primary biochemical AD classification').columns)
        features_list = list(df.columns)
        # default_features = ["Q13228;Q13228-4", "P14618", "Q6EMK4"]
        default_features = ['P04075', 'P14618-2', 'P08294', 'P62937', 'Q6EMK4', 'P17174', 'H9KV31;O15394', 'P10451-5']
        text_feature_value = '\n'.join(default_features)
    elif (uploaded_file is None) & ('Immune' in default_filename):
        ldc = LoadDatasetCheck(default_filename, targetvalue_column=None).load_dataset(set_index_col=None,
                                                                                       dropna_row=False)
        load_df = ldc.df.copy(deep=True)
        df = ldc.df.copy(deep=True)
        sample_identifier_option = list(df.columns)
        targetvalue_column_option = ['Response (1:Responder; 0:Non-responder)'] + list(
            df.drop(columns='Response (1:Responder; 0:Non-responder)').columns)
        features_list = list(df.columns)
        default_features = ["Cancer_Type2", "Chemo_before_IO (1:Yes; 0:No)", "Age", "Sex (1:Male; 0:Female)", "BMI", "Stage (1:IV; 0:I-III)", "NLR", "Platelets", "HGB", "Albumin", "Drug (1:Combo; 0:PD1/PDL1orCTLA4)", "TMB", "FCNA", "HED", "HLA_LOH", "MSI (1:Unstable; 0:Stable_Indeterminate)"]
        text_feature_value = '\n'.join(default_features)
    elif (uploaded_file is None) & ('Pan-cancer' in default_filename):
        ldc = LoadDatasetCheck(default_filename, targetvalue_column=None).load_dataset(set_index_col=None,
                                                                                       dropna_row=False)
        load_df = ldc.df.copy(deep=True)
        df = ldc.df.copy(deep=True)
        sample_identifier_option = list(df.columns)
        targetvalue_column_option = ['Cancer'] + list(
            df.drop(columns='Cancer').columns)
        features_list = list(df.columns)
        default_features = ["TCL1A", "STC1", "FCRL2", "CD22", "FCER2", "CD6", "GLO1", "CHRDL2", "FCGR3B", "CRNN", "AGER", "MFAP5", "LYPD3", "DNER", "IL20", "FAP", "CXCL6", "CD34", "CDH17", "CRTAC1", "IL18RAP", "TRAF2", "ADAMTS8", "GZMB", "SPINK5", "F3", "ICAM4", "PRTG", "LAMP3", "SDC4", "OXT", "HSD11B1", "BTC", "LPL", "MSMB", "PLAT", "TNFSF10", "DPT", "CLEC7A", "CLMP", "AFP", "PAEP", "CDH3", "SSC5D", "PRDX5", "TFRC", "PADI4", "FKBP1B", "PRDX6", "LGALS4", "CCL20", "SELE", "LAP3", "AREG", "CEACAM5", "MMP12", "BCL2L11", "LSM1", "BPIFB1", "ABHD14B", "CXCL17", "MLN", "ANXA11", "SFTPD", "MTPN", "SCGB3A2", "LBP", "ACP5", "TFPI2", "COL9A1", "CNTN5", "SLAMF7", "MZB1", "GFAP", "BCAN", "ADAMTS13", "CD244", "FLT3", "TNFSF13B", "CXCL9", "CXCL13", "DCXR", "SERPINA9"]
        text_feature_value = '\n'.join(default_features)
    show_data = False
    if not df.empty:
        st.success(body='✅ File loaded successfully!')
        show_data = st.checkbox("data preview")

#
if show_data:
    st.header("1. Upload Dataset")
    if (uploaded_file is None) & (default_filename is not None):
        if 'Gestational' in default_filename:
            st.info("""
                        Liang *et al* (2020) utilized this dataset to predict gestational age and time to delivery in pregnant women. This dataset was used by OML as an example dataset for regression modeling.\n
                        **Reference**\n
                        Liang et al., 2020, Cell 181, 1680–1692 2020. Published by Elsevier Inc. https://doi.org/10.1016/j.cell.2020.05.002.
                        """)
        elif 'Alzheimer' in default_filename:
            st.info("""
            The Alzheimer dataset was obtained from the Bader *et al* (2020). OmicLearn  platform (2022) curated the dataset as an example dataset for its platform. The OML platform used the dataset that was curated by OmicLearn to demonstrate the process of building machine learning model.\n
            **Reference**\n
            Bader, Jakob M., et al. "Proteome profiling in cerebrospinal fluid reveals novel biomarkers of Alzheimer's disease." Molecular systems biology 16.6 (2020): e9356. https://doi.org/10.15252/msb.20199356\n
            Torun, Furkan M., et al. "Transparent exploration of machine learning for biomarker discovery from proteomics and omics data." Journal of Proteome Research 22.2 (2022): 359-367.  https://doi.org/10.1021/acs.jproteome.2c00473\n
            """)
        elif 'Immune' in default_filename:
            st.info("""
            Diego Chowell *et al* (2022) utilized this dataset to develop a machine learning model to predict ICB response by integrating genomic, molecular, demographic and clinical data from a comprehensively curated cohort (MSK-IMPACT) with 1,479 patients treated with ICB across 16 different cancer types. This dataset was used by OML as an example dataset for classification modeling.\n
            **Reference**\n
            Chowell D, Yoo SK, Valero C, et al. Improved prediction of immune checkpoint blockade efficacy across multiple cancer types. Nat Biotechnol. 2022;40(4):499-506. doi:10.1038/s41587-021-01070-8.
            """)
        elif 'Pan-cancer' in default_filename:
            st.info("""
            The pan-cancer dataset imported as an example data in OML was obtained from the U-CAN dataset, comprising 1477 patients from 12 cancer types. Álvez MB, *et al* (2023) utilized this dataset to develop a **Multiclass Prediction Model** using 83 proteins for pan-cancer. The normalized U-CAN dataset containing NPX values can be downloaded from BioStudies database under accession code S-BSST935. We processed the downloaded dataset in 2 steps: 1 The wild format that each row represents one patient sample for NPX values was created from the long format of raw data; 2 Missing values were predicted by KNN algorithm using sklearn.impute.KNNImputer function.\n
            **Reference**\n
            Álvez MB, Edfors F, von Feilitzen K, et al. Next generation pan-cancer blood proteome profiling using proximity extension assay. Nat Commun. 2023;14(1):4308. Published 2023 Jul 18. doi:10.1038/s41467-023-39765-y.\n
            Linn Fagerberg Linn Fagerberg E-mail: linn.fagerberg@scilifelab.se ORCID: 0000-0003-0198-7137 Affiliation: Royal Institute of Technology , Linn Fagerberg, María Bueno Álvez María Bueno Álvez E-mail: maria.bueno@scilifelab.se Affiliation: Royal Institute of Technology , María Bueno Álvez, Linn Fagerberg Linn Fagerberg E-mail: linnfa@kth.se ORCID: 0000-0003-0198-7137 Affiliation: Royal Institute of Technology & Linn Fagerberg (2023). Next generation pan-cancer blood proteome profiling using proximity extension assay. BioStudies, S-BSST935. Retrieved from https://www.ebi.ac.uk/biostudies/studies/S-BSST935.
            """)
    st.markdown(f"""The dimension of dataset is **`{load_df.shape[0]}`** rows and **`{load_df.shape[1]}`** columns. """)
    AgGrid(load_df, height=400)
    load_df_csv = convert_df(load_df)
    col1, col2 = st.columns((2, 1))
    with col2:
        st.download_button(data=load_df_csv, file_name="1_upload_data.csv", label="⬇️ Download upload data (.csv)",
                           mime='text/csv', )

with st.sidebar.expander(modules["module_2"], expanded=False):
    st.info("""
    **Target variable** is also sometimes called the dependent variable, criterion variable or outcome variable.\n
    **Independent Variable** is also known as explanatory or predictor variable. In machine learning, independent variable can be called **Feature**.
    """)
    sample_identifier_help = """We recommend that users specify a column name as the identifier for each sample. If the "None" option is chosen, integers from the range of 0 to the number of samples will be used as the identifiers."""
    sample_identifier = st.selectbox(label='Sample identifier',
                                options=sample_identifier_option,
                                help=sample_identifier_help)

    if (sample_identifier != 'None') & (not df.empty):
        df.set_index(sample_identifier, inplace=True)
        targetvalue_column_option.remove(sample_identifier)
        features_list.remove(sample_identifier)
        sample_identifier_value = df.index.tolist()
    else:
        sample_identifier_value = list(range(0, df.shape[0]))

    targetvalue_column_info = "Target variable( or dependent variable or response variable) is the output labels of input varibales."
    targetvalue_column = st.selectbox('Target variable',
                                      targetvalue_column_option,
                                      help=targetvalue_column_info)
    if targetvalue_column == 'None':
        targetvalue_column = None
    if (not df.empty) & (targetvalue_column is not None):

        load_rows = df.shape[0]
        load_cols = df.shape[1]
        df = df[df[targetvalue_column].notna()]
        # reset sample_identifier_value
        sample_identifier_value = df.index.tolist()
        update_rows = df.shape[0]
        if load_rows != update_rows:
            st.warning(
                f"Found  {load_rows - update_rows} missing value in {targetvalue_column}. The missing value will be removed in downstream pipeline!!!")
        df = df.dropna(axis='columns')
        update_cols = df.shape[1]
        if update_cols != load_cols:
            st.warning(
                f"Drop {load_cols - update_cols} columns (or independent variables) that contains at least one missing values")

    features_list = list(df.columns)
    if targetvalue_column is not None:
        features_list.remove(targetvalue_column)

    targetvalue_type_info = """
    Each variable will be one of two types: **categorical** or **numerical**.\n
    **Classification**: The target variable must be **categorical** data.For example, it could be binary, such as whether a patient is cancer (yes or no), or multiclass, such as the type of cancer subtype. Generally, the initial categorical variable can be represented using characters, but in downstream analysis, it needs to be converted to numerical representation, for example, 0 represents control and 1 represents cancer. Note that if numerical variables have been used to replace categorical variables in user data, the target variable type is still categorical, and the downstream module's default algorithm type is classification.\n
    **Regression**: The target variable must be **numerical** data, such as height. The downstream module's default algorithm type is regression.
    """

    targetvalue_type_option = ['None', 'Numerical', 'Categorical']
    if (not df.empty) & (targetvalue_column is not None):
        targetvalue_type_option = ['Numerical', 'Categorical']
        if (is_object_dtype(df[targetvalue_column])) | (is_integer_dtype(df[targetvalue_column])):
            targetvalue_type_option = ['Categorical', 'Numerical']


    targetvalue_type = st.selectbox('Categorical or Numerical for target variable',
                                    targetvalue_type_option,
                                    help=targetvalue_type_info)
    if targetvalue_type == 'Categorical':
        targetvalue_ele = list(set(df[targetvalue_column]))
        targetvalue_ele.sort()
        targetvalue_ele_options = []
        targetvalue_ele_options.append(list(range(0, len(targetvalue_ele))))
        ele_dummys = dict()
        for i_ele, ele in enumerate(targetvalue_ele):
            targetvalue_type_dummy = st.selectbox(f" Class {ele}: assign a unique integer",
                                                  targetvalue_ele_options[i_ele])
            ele_dummys[ele] = targetvalue_type_dummy
            targetvalue_ele_options.append([i for i in targetvalue_ele_options[i_ele] if i != targetvalue_type_dummy])
        df[targetvalue_column].replace(ele_dummys, inplace=True)

    features_list = df[features_list].select_dtypes(include=np.number).columns.tolist()
    truncate_feature_name = {x: f"{x[0:200]}<truncate>" if len(x) > 200 else x for x in features_list}
    df = df.rename(columns=truncate_feature_name)
    features_list = list(truncate_feature_name.values())

    all_features_comment = st.text_area("All independent variables (must be Numerical)", '\n'.join(features_list),
                           help=""" 
                           In omics research, all independent variables can encompass all omics molecules detected in the experiment, including **proteins**, **metabolites**, and **lipids**, among others.\n
                           Note: \n
                           1. **Each row** represents one independent variable or feature. Please do not add any punctuation to prefix or suffix words.\n
                           2. The input feature name must match the column name of the uploaded data's feature, otherwise an error will occur.\n""")
    if all_features_comment == "":
        features = []
    else:
        features = all_features_comment.split('\n')
        features = list(set(features))
        st.warning(
            """⚠️ Warning: By default, all non-missing and numerical variables (or features) will be shown except for the target variable. The user can remove any unused variables from the list up as desired.\n""")

show_trandata = False
with st.sidebar.expander(modules["module_3"], expanded=False):

    if (targetvalue_column is not None) & (features != []):
        select_df = df[[targetvalue_column] + features]
        select_df = select_df.dropna()
    else:
        select_df = pd.DataFrame()

    #
    trm_info = """
        **This option is specific to dependent variable of the regression**.\n
        """
    if targetvalue_type == 'Numerical':
        targetvalue_trm = st.selectbox('Transform target value',
                                       ['None', 'Z-Score', 'LOG', 'LOG2', 'LOG10', 'SQRT', 'RECIPROCAL', 'SQUARE',
                                        'LOGIT'],
                                       help=trm_info)
    else:
        targetvalue_trm = 'None'

    if (targetvalue_trm != 'None') & (targetvalue_column != 'None'):
        if not select_df[targetvalue_column].empty:
            select_df = feature_trans(select_df, feature=targetvalue_column, feature_method=targetvalue_trm)

    features_trm = st.selectbox('Transform features',
                                ['None', 'LOG2', 'Z-Score', 'LOG', 'LOG10', 'SQRT', 'RECIPROCAL', 'SQUARE', 'LOGIT'],
                                help="""All selected features will be normalized by the same method.""")
    if (features_trm != 'None') & (len(features) > 0):
        if not select_df[features].empty:
            select_df = feature_trans(select_df, feature=features, feature_method=features_trm)

    if not select_df.empty:
        if (features_trm == 'None') & (targetvalue_trm == 'None'):
            st.warning("⚠️ Warning: No normalization method chosen by user!!!")
        else:
            st.success('✅ Data normalized successfully')

        show_trandata = st.checkbox("Normalized data preview")


if show_trandata:
    st.header("3. Normalize Dataset")
    st.markdown(f"""The shape of dataset:  **`{select_df.shape[0]}`** rows, **`{select_df.shape[1]}`** columns.""")
    trandata_df = copy.deepcopy(select_df)
    trandata_df.insert(loc=0, column='Sample Identifier', value=sample_identifier_value)
    AgGrid(trandata_df, height=400)
    csv = convert_df(select_df)
    col1, col2 = st.columns((2, 1))
    with col2:
        st.download_button(data=csv, file_name="3_normalized_data.csv", label="⬇️ Download normalized data (.csv)",mime='text/csv',)


show_all_features_importance = False
with st.sidebar.expander(modules["module_4"], expanded=False):
    st.info('This module refers to the process of selecting the most important features in a model based on their ranking or importance. This ranking or importance is often done using techniques like feature importance scores, coefficients, or feature weights assigned by the model. \n')
    if targetvalue_type == 'Numerical':
        targetvalue_type_option = ['Supervised Regression', 'Supervised Classification']
    elif targetvalue_type == 'Categorical':
        targetvalue_type_option = ['Supervised Classification', 'Supervised Regression']
    else:
        targetvalue_type_option = ['None', 'Supervised Classification', 'Supervised Regression']

    type_ml = st.selectbox("Select ML Type", targetvalue_type_option,
                     help="""
                     For **classification**: the feature importance of the selected features was calculated using both **Logistic Regression and Random Forest Models**. The resulting rankings were then compared by plotting a scatter plot, which shows the relationship between the feature importances obtained by the two methods.
                     """)
    run_features_importance = st.checkbox('Click to run ')


if run_features_importance:
    scatter_feature_im, scatter_feature_importance_fig, x_model, y_model = \
        feature_select_from_model_table(type_ml, targetvalue_column, select_df, features)
    st.header("4. Feature Selection from Model")
    st.markdown(f"""
    Compared to basic statistical methods for feature selection, `Feature Selection from Model` is more challenging for users without programming background because it requires programming skills and understanding of model fundamentals. Therefore, we offer the `Feature Selection from Model` module for preliminary feature selection. Mathias Uhlen et al (2022) utilized linear model and non-linear model to examine the distribution of feature importance for all proteins in the Olink panel and chose the proteins most pertinent to pan-cancer for downstream modeling according to the findings.\n
    Here, for **`{type_ml}`** defined by user, the feature importances of the selected features were calculated using both **`{x_model}` and `{y_model}`**. The resulting rankings were then compared by a scatter plot, which shows the relationship between the feature importances obtained by the two methods.\n
    **`{x_model}`** is regarded as a linear model, while **`{y_model}`** is regarded as a non-linear mode. The purpose of using both linear and nonlinear models is to observe the importance distribution of all features in the two models. Users can try to select the most important features for downstream modeling steps. Note that this is only one of the feature pre-selection methods, and there are many factors that can affect the results, such as the correlation between features. Therefore, sometimes low feature scores do not necessarily indicate that the feature is unimportant. \n
    **Reference**\n
    [1] Uhlen, Mathias, et al. "Next generation pan-cancer blood proteome profiling using proximity extension assay." DOI: https://doi.org/10.21203/rs.3.rs-2025767/v1 (2022).
    [2] Álvez MB, Edfors F, von Feilitzen K, et al. Next generation pan-cancer blood proteome profiling using proximity extension assay. Nat Commun. 2023;14(1):4308. Published 2023 Jul 18. doi:10.1038/s41467-023-39765-y.
    """)
    st.subheader('4.1 The ranking table')
    st.markdown(f"""
    The higher the scores in the ranking column, i.e. **`Rank of {x_model}`** or **`Rank of {y_model}`**, the more important it is in the model.\n
    """)
    AgGrid(scatter_feature_im, height=400)
    scatter_feature_im_csv = convert_df(scatter_feature_im)
    col1, col2 = st.columns((2, 1))
    with col2:
         st.download_button(data=scatter_feature_im_csv, file_name="4.1_feature_ranking_table.csv", label="⬇️ Download the ranking table (.csv)",
                           mime='text/csv', )

    st.subheader('4.2 The Scatter plot of feature importance')
    st.plotly_chart(scatter_feature_importance_fig)

with st.sidebar.expander(modules["module_5"], expanded=False):
    st.info("""
    This module requires the user to input important features, where the number of important features is limited to the total number of features. Important features can be pre-defined by the user or identified by the **Feature Selection from Model** module as highly ranked features, or selected by other feature selection methods.
    """)
    # st.sidebar.markdown("# Normalize Dataset")
    input_feature_method_help = """
            **multi-selects**: prefer for small number of features\n
            **input text**: prefer for large number of features\n
            In machine learning and pattern recognition, **a feature is an individual measurable property or characteristic of a phenomenon** (from wiki).
            In bioinformatics, proteins, metabolites, lipids, clinical indices and so on can be considered as features for ML models.
            """
    input_feature_method = st.radio("Choose menu for input features",
                                            ["multi-selects", "input text comment"],
                                            help=input_feature_method_help)

    if input_feature_method == "multi-selects":
        im_features = st.multiselect('Features (multi-selects)',
                                          features_list,
                                          default=default_features,
                                          help='choose multi-features')
    elif input_feature_method == "input text comment":
        comment = st.text_area("Features:", text_feature_value, help="""
                **each row** represents one feature, don't add any punctuation in prefix-word or suffix-word
                """)
        if comment == "":
            im_features = []
        else:
            im_features = comment.split('\n')
            im_features = list(set(im_features))

    if (targetvalue_column is not None) & (len(im_features) == 0):
        st.error('❌ Some Independent Variables (or Features) must be provided!')

    if not select_df.empty:
        ml_x = select_df[im_features]
        y = select_df[targetvalue_column]
    else:
        ml_x = None
        y = None


with st.sidebar.expander(modules["module_6"], expanded=False):
    # st.sidebar.markdown('# Split Data into Training Set and Testing Set')
    cv_help = """
                If dataset is small, **cross-validation** is recommended to split dataset. You can also select **train_test_split** method to split data when data is large.
                * **cross-validation**: set three parameters containing n-fold, n-repeats and cross validation method
                * **train_test_split**: set the proportion of test set size
                * **select the specific Sample Identifier for testing set**: set [start sample identifier, end sample identifier] for testing set, and the remaining rows treated as training set.
                * Note: **RepeatedStratifiedKFold** of CV only used in classification
                """
    split_methods = ['Cross Validation', '% Training Set and % Testing Set', 'select the specific Sample Identifier for testing set (others treated as training set)']
    split_method = st.radio('Choose method for splitting data', split_methods, help=cv_help)
    test_size = None
    sample_identifier_dset = None
    ml_x_train = None
    y_train = None
    ml_x_independ_test = None
    y_independ_test = None
    if split_method == 'Cross Validation':
        cv_method_instruction = """
        see more details about Cross-validation, please click [Wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
        """
        if type_ml == 'Supervised Regression':
            cv_method_option = ['RepeatedKFold',  "LeaveOneOut", "LeavePOut"]
        else:
            cv_method_option = ['RepeatedStratifiedKFold', 'RepeatedKFold', "LeaveOneOut", "LeavePOut"]
        cv_method = st.selectbox('1. CV method', cv_method_option, help=cv_method_instruction)
        # default values
        n_fold = 2
        n_repeat = 2
        random_state = 42
        leavep = 2
        if cv_method in ['RepeatedStratifiedKFold', 'RepeatedKFold']:
            n_fold = st.number_input('2. Number of folds', 2, 100, 2,
                                             help="K-Fold CV is where a given data set is split into a K number of sections/folds where each fold is used as a testing set at some point.")
            n_repeat = st.number_input('3. Number of times', 1, 100, 2,
                                               help="repeating cross-validation multiple times where in each repetition, the folds are split in a different way")
            random_state = st.slider('4. Controls the generation of the random states', 0, 137, 42,
                                             help="""
                                      Controls the randomness of each repeated cross-validation instance. Pass an int for **reproducible output** across multiple function calls ([source](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html?highlight=repeatedkfold#sklearn.model_selection.RepeatedKFold))
                                      """)
        elif cv_method == "LeavePOut":
            leavep = st.number_input("Size of the test sets", 1, 500, 2,
                                             help="""
                                      **Must be strictly less than the number of samples**.
                                      For large dataset, this method can be very costly.
                                      Click [leavepout of scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePOut.html?highlight=leavepout#sklearn.model_selection.LeavePOut)
                                      """)

        cv = select_cross_validation(cv_method=cv_method, n_splits=n_fold, n_repeats=n_repeat,
                                     random_state=random_state, leavep=leavep)
        if (ml_x is not None) & (y is not None):
            cv_split = [(train_index, test_index) for train_index, test_index in cv.split(ml_x, y)]
            ml_x_train = ml_x
            y_train = y
            ml_x_independ_test = None
            y_independ_test = None
            sample_identifier_dset = pd.DataFrame({'Sample Identifier': ml_x_train.index,
                                                   'cv_index': range(0, ml_x_train.shape[0])})
            sample_identifier_dset.set_index('cv_index', inplace=True)

    elif split_method == '% Training Set and % Testing Set':
        stratify = None
        if type_ml == 'Supervised Classification':
            use_stratify = st.checkbox("Stratify (Preserve the class ratios in both train and test dataset)",
                                       value="Stratify (Preserve the class ratios in both train and test dataset)")
            if use_stratify:
                stratify = y

        test_size_help = """
        The proportion should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        Click [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train_test_split#sklearn.model_selection.train_test_split) to see more information about this method.
        """
        test_size = st.slider('the proportion of testsize', 0.0, 1.0, 0.2, help=test_size_help)
        random_state = st.slider('Controls the generation of the random states', 0, 137, 42)
        if (ml_x is not None) & (y is not None):
            ml_x_train, ml_x_independ_test, y_train, y_independ_test = train_test_split(ml_x, y, test_size=test_size,
                                                                                        random_state=random_state, stratify=stratify)
            cv_split = None
    elif split_method == 'select the specific Sample Identifier for testing set (others treated as training set)':
        start_row = st.selectbox("start sample identifier (include) in testing set", ['None'] + list(ml_x.index))
        end_row = st.selectbox("end sample identifier (include) in testing set", ['None'] + list(ml_x.index))
        if (start_row == end_row) | (start_row == "None") | (end_row == "None"):
            st.warning("⚠️ Warning: start sample identifier must not equal end sample identifier")
        else:
            ml_x_independ_test = ml_x.loc[start_row:end_row, :]
            y_independ_test = y.loc[start_row:end_row]
            test_size = np.round(ml_x_independ_test.shape[0] / ml_x.shape[0], 2)

            train_rows = [x for x in ml_x.index if x not in ml_x_independ_test.index]
            ml_x_train = ml_x.loc[train_rows, :]
            y_train = y.loc[train_rows]
            cv_split = None
            # st.dataframe(y_independ_test)
            # st.dataframe(y_train)


with st.sidebar.expander(modules["module_7"], expanded=False):
    reg_methods = ['Linear Regression', 'Ridge Regression', 'Lasso Regression',
                    'ElasticNet Regression', 'Bayesian Ridge Regression', 'Support Vector Regression',
                    'K-Nearest Neighbors Regression', 'Random Forest Regressor']
    clf_methods = ['Logistic Regression', 'Support Vector Classification', 'Gaussian Naive Bayes', 'K-Nearest Neighbors',
                   'Random Forest Classification', 'AdaBoost', 'GradientBoosting', 'Xgboost', 'LightGBM', 'CatBoost']
    if type_ml == 'Supervised Regression':
        ml_method = st.radio('Select a regression algorithm', reg_methods)
    elif type_ml == 'Supervised Classification':
        ml_method = st.radio('Select a classification algorithm', clf_methods)
    else:
        ml_method = None
        show_ml_type = st.radio('ML algorithm', ['Supervised Regression', 'Supervised Classification'], help="Click the button to display the algorithms included in the platform.")
        if show_ml_type == 'Supervised Regression':
            reg_methods_info = ', '.join(reg_methods)
            st.markdown(f"""**{reg_methods_info}**""")
        else:
            clf_methods_info = ', '.join(clf_methods)
            st.markdown(f"""**{clf_methods_info}**""")

pro = "Undo"


submitted = False

if (ml_x_train is not None) & (len(im_features) > 0):
    st.header("7. Choose Machine Learning Method")
    st.subheader(f"👇 Please set the `{ml_method}` model parameters, then click the submit button")
    form = st.form(key="annotation")
    with form:
        set_model_parameters, model = ml_model_set_paras(ml_method)
        submitted = form.form_submit_button(label="Submit")

if submitted:
    latest_iteration = st.empty()
    # st.info("It takes a few minutes. Please wait.")
    my_bar = st.progress(0)
    for percent_complete in range(100):
        if percent_complete == 20:
            mlcv_class_regression, output_performance_model = \
                ml_model_run(type_ml, ml_method, model, ml_x_train, y_train, ml_x_independ_test,
                             y_independ_test, set_model_parameters, cv_split, im_features)
        if percent_complete == 80:
            ml_model_show_figure(mlcv_class_regression, type_ml, ml_method, ml_x_train, y_train,
                                 ml_x_independ_test,
                                 cv_split, targetvalue_column, sample_identifier_dset, test_size)
        if percent_complete == 99:
            ml_model_show_summary_table(type_ml, cv_split, ml_method, output_performance_model)

        my_bar.progress(percent_complete + 1)
        latest_iteration.text(f'{percent_complete + 1}% of analysis completed')


st.sidebar.write("***")
st.sidebar.markdown("""
**:red[Contact Us]**\n
**amy.zhang@qlife-lab.com**
""")
st.sidebar.markdown("""
**:red[Other useful tools for Omics data]**\n
**[Omics Batch Correct](https://omia.untangledbio.com/obc/)**\n
**[Omic Integrate Analysis](https://omia.untangledbio.com/oia/)**
""")