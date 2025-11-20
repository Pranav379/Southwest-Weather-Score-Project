# LightGBM Results
## Plots
### MAE
![LightGBM MAE Plot](LightGBM_Plots/LG_TT_MAE.png)
### RMSE
![LightGBM RMSE Plot](LightGBM_Plots/LG_TT_RMSE.png)
### R^2
![LightGBM R^2 Plot](LightGBM_Plots/LG_TT_R2.png)
### Feature Importances
![LightGBM Feature Importances](LightGBM_Plots/LG_Features.png)
## Interpretations
The LightGBM model exhibited worse performance overall on the test set compared to the train set, and produced a negative R^2 value on the test set, indicating that the model performed worse than a naive mean forecast.

Interestingly, pressure dominated over other weather factors such as wind speed, precipitation, and snow in the model. Departure time and distance had a greater impact on the model overall than weather factors, suggesting that weather conditions have less effect on flight delays than logistical factors.
# Random Forest Results
## Plots
### MAE
![Random Forest MAE Plot](RandomForest_Plots/RF_TT_MAE.png)
### RMSE
![Random Forest RMSE Plot](RandomForest_Plots/RF_TT_RMSE.png)
### R^2
![Random Forest R^2 Plot](RandomForest_Plots/RF_TT_R2.png)
### Feature Importances
![Random Forest Feature Importances](RandomForest_Plots/RF_Features.png)
## Interpretations
Similar to the LightGBM model, the Random Forest model performed worse overall on the test set compared to the train set, but outperformed the LightGBM model in each category. The Random Forest model reached a maximum R^2 of about .15, reflecting the overall poor performance of ensemble learning models. Weather factors had greater importance on the Random Forest model, but logistical variable such as departure time and distance still ranked towards the top.
# Generalized Model Results
## Plots
### Cancellations
![Generalized Model Cancellations](Generalized_Plots/Cancellation.png)
### Score Graph
![Generalized Model Score Graph](Generalized_Plots/Score.png)
## Interpretations
The model employs Min-Max scaling to normalize a log-scaled delay score to a 0-100 range. Flights that were cancelled are assigned a score of 100.