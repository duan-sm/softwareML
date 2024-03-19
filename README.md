
This is about intelligent model building software front-end and back-end code.

The front-end includes all of .ui files, and .py file that is same name as .ui.

The back-end includes the rest .py files.

The folder data_pressure represents the pressure data.

The folder example data represents the stress data

The folder helper is an introduction to the algorithms needed to run the software.

The folder res contains the output of the build model


# 1 Front-end

The relationships of five page are as follows:
![img.png](img.png)

ReadData will get data and visualize it.

Outlier will select outliers in data, delete them and visualize them.

FeatureAnalysis will compute the relation of features and visualize it. Besides, we can normalize the data in this page.

Model will build intelligence model, then train and verify it.

Test will test the training model and visualize the results.

## 1.1 ReadData

There are 6 functions. The relationship between buttons and functions is as follows (Italics represent buttons): 
LoadData: Back-end function, "DateInputBtn", It will open the file selection screen, read the selected file and display it.

Stress/Pressure: Input data type

Region: Select visualization area for plotting

DeleteSpaceR: Back-end function, " btnDelRows", It will delete empty row in data.

DeleteSpaceC: Back-end function, " btnDelCols", It will delete empty column in data.

SaveData: Back-end function, " btnSaveData", The corrected data presentation will be modified

WholePipe-CentreLine: Back-end function, "DrawFig", Draws a slice of the selected location (as determined by the input page data). 

DarwSlice: Back-end function, " DrawWholePipe ", Draw the center curve of the pipe.

## 1.2 Outlier

There are 3 functions. The relationship between buttons and functions is as follows (Italics represent buttons): 

SelectOutlier: Back-end function, " Outlier", The modified 3 sigma is used to determine outliers for each feature in the data.

Draw: Back-end function, "OutlierDraw", Plots values and outliers for selected features.

SaveData: Back-end function, " SaveDataBtn", The corrected data presentation will be modified

ReturnReadData: Return to ReadData page.

## 1.3 FeatureAnalysis

There are 6 functions. The relationship between buttons and functions is as follows (Italics represent buttons): 

CorrelationAnalysis: Back-end function, " CorrDraw", Pearson correlation coefficient and Spearman correlation coefficient are used to calculate the correlation coefficient between each feature.

Chi-square test: Back-end function, " chi2Btn", Chi-square test is used to analyze the relationship between features.

Mutual information test: Back-end function, " mutualInforBtn", The relationship between features is analyzed by mutual information method.

Normalization: Back-end function, " NondimenBtn", The data is processed without dimension. It can be divided into standardization method and maximum and minimum method.

DimensionalityReduction: Back-end function, " DimenComBtn ", The principal component analysis is used to process the data and reduce the data dimension. (Processed data will lose its physical meaning).

SaveData: Back-end function, " SaveData", The corrected data presentation will be modified

ReturnReadData: Return to ReadData page.

## 1.4 Model

There are 2 functions. The relationship between buttons and functions is as follows (Italics represent buttons): 

ANN/LSTM: Select type of model for training.

Initialize: Back-end function, " Initialize", Since there are many parameters of the intelligent model, the parameters of the model can be quickly initialized through this function (the initial parameters are only for reference and are not necessarily applicable to the current data).

Compute: Back-end function, " Compute", The data is divided into validation set and training set by 7:3. The model is trained and verified according to the set parameters.

## 1.5 Test

There are 2 functions. The relationship between buttons and functions is as follows (Italics represent buttons): 

LoadTestData: Back-end function, " LoadTestData", Select the file window. Select the data to test the model against.

LoadPara: Back-end function, " LoadParaBtn", Select the file window. Select model normalization parameters. See TestDataPara framework for slice location selection.

SelectModel: Select the file window. Select the model to be tested.

Test: Back-end function, " Test", Test button. When the test data is imported and the normalized parameters and corresponding model are selected, the second button can be used to test the model.

DrawSliceT: Back-end function, " DrawSliceT", Draw the simulation results of slices and the prediction results of intelligent models. See TestDataPara framework for slice location selection.

# 2 Back-end

software_IntelligenceModel.py represents the front-end and back-end contact files.

_ReadData.py is the function required in the ReadData interface.

_Outlier.py is the function required in the Outlier interface.

_FeatureAnalysis.py is a function required in the FeatureAnalysis interface.

_Model.py is the required function in the Model interface.

_Test.py is the function required in the Test interface.