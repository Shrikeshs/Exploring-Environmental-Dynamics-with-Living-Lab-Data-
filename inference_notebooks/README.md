# Inference of Features
## Correlation
The correlation values offer significant insights into the relationships between PM2.5 and other variables in the dataset.
![image](https://github.com/user-attachments/assets/a244cbc0-379b-4912-a690-0357495a60fd)
Figure. Correlation of features with PM2.5

PM2.5 ->	1

PC2.5 ->	0.95061

PC1.0	-> 0.800025

PM1.0	-> 0.797749

PM0.1	-> 0.748332

PC0.1 -> 	0.748304

Timestamp	-> 0.695188

PM5.0 ->	0.677009

PM0.3 ->	0.577859

PC0.3 ->	0.574874

PC5.0 ->	0.551336

PM10.0	0.467966

PC10.0 ->	0.443244

CO2 ->	0.399983

Humidity AQ ->	0.29754

PM0.5 ->	0.168491

Pressure AQ ->	0.160372

VOC AQ ->	0.117744

No2 Gas ->	0.000015

Solar Panel Power ->	0.007095

PC0.5 ->	0.034362

NOX AQ ->	0.034448

Temp AQ ->	0.283784

Table. Correlation values of features relative to the target variable

The above correlation matrix represents the relationships between our target variable PM2.5 and other features. Here is an interpretation of the results: 
- 	**PC2.5**: Shows a strong positive correlation to PM2.5 which makes sense as it likely represents the partial count of particulate matter of below 2.5 micrometres.
- 	**PC1.0 (0.800025) and PM1.0 (0.797749)**: Both show strong positive correlation with PM2.5 thereby suggesting that increase in particulate matter of 1.0 micrometres will result in increase in PM2.5 levels in the air.
-   **PM0.1 (0.748332) and PC0.1 (0.748304)**: Both show moderately strong positive correlation with PM2.5 suggesting that increase in the fine matter of 0.1 micrometres will result in the increase in the PM2.5 levels.
-   **PM5.0 (0.677009) and PM0.3 (0.577859)**: Larger particles seem to be lesser correlated to PM2.5 levels.
-   **CO2 (0.399983) and Humidity AQ (0.297540)**:  Thes two features have a weak positive correlation to PM2.5 suggesting their proportionality. It infers the slight dependence of PM2.5 values with the CO2 and Humidity levels.
- 	**Temp AQ (-0.283784)**: Temperature of the condition at the time of sensor recording seems to be negatively fairly correlated to PM2.5 values.  As temperature increases, PM2.5 values seem to be slightly reduced.

## Trend Analysis
In this section, we inspect how PM2.5 values act with certain picked out parameters based on the above correlation analysis. This trend analysis allows us to compare how different features fair against time. 
### Seasonal Trend: 
In this section, we compare the seasonal trend of PM2.5 values against the months of the sensor recording.
 ![image](https://github.com/user-attachments/assets/ef7eed21-2acd-4718-8e66-c62b82811625)

Figure . Seasonal Trend of PM2.5
The following conclusions can be traced from the above graph:
- High PM2.5 values during the early months of the year (January – May)
  * During these months, since, Leeds UK experiences colder months, the heating in all homes is at a high. This research conducted an intensive experiment to affirm that PM2.5 levels are considerably increase during the winter seasons due to high household heating. (Olszowski, 2019)

  * Cold weather results in lesser air currents meaning the pollutants can accumulate more.
- 	Sharp decline in June
  * Due to the onset of summer, warmer temperatures result in reduction of PM2.5 values.Thereby deducing that the increase in temperature results in the reduction of pollutants.
- Low PM2.5 levels from July to September
*	Due to increased sunlight during these months, the air is clearer, and the pollutants are broken down.
-	Increasing trend of PM2.5 from November-December
*	As the year progresses, due to the onset of winter in the UK, the levels gradually increase.
*	According to this study, the famous Guy Fawkes night in the UK, which is celebrated on the 5th of November, the average PM2.5 levels are about 31.25-38.75. That’s about 2-3-fold increase in PM2.5 levels. (Singh, Pant, & Pope, 2024)
 ![image](https://github.com/user-attachments/assets/4bce81aa-e0cb-43ac-b505-98cc099c8bd4)

Figure. Scatter plot of Temperature vs PM2.5 levels
The above scatter plot of Temperature vs PM2.5 values generated from our dataset proves the same inference. The lower temperatures between 0 to 25 seem have high PM2.5 values.
## OLS Regression
We use OLS Regression to explore the relationship between PM2.5 and chosen variables like Temperature, Humidity AQ, CO2 and No2 Gas values. 
This allows us to determine if these variables, which is kept as a constant, help in the variance of PM2.5 values for our dataset.
  ![image](https://github.com/user-attachments/assets/fd32678b-b869-450a-8d68-eec032f018d0)

Figure . OLS Regression Results
The above results show an OLS regression model with the following parameters:
Dependent variables: CO2 gas, PM1.0 and PM0.1
o	Independent variables: PM2.5
The following trends and analysis can be retrieved from the results above
-	R-squared value of 0.700 means about 70 percent of the model’s variance is explained by the three variables in question.
-	Prod(F-statistic) value of 0.00 which typically indicates that the model is statistically significant.
-	The coef values can be interpreted as follows
o	PM0.1 is 3590.9406 which means that each additional unit increase in PM0.1, PM2.5 can be expected to increase 3590.9406 units keeping all other variables constant.
