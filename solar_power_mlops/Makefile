.PHONY: prepare-data build-features split-data train-model predict plot

PLANT_ID ?= 2

prepare-data:
	python spmlops/dataset.py \
		--input-plant-path data/raw/Plant_$(PLANT_ID)_Generation_Data.csv \
		--input-weather-path data/raw/Plant_$(PLANT_ID)_Weather_Sensor_Data.csv \
		--output-path data/processed/Processed_Plant_$(PLANT_ID)_Generation_Data.csv

build-features:
	python spmlops/features.py build-features \
		--input-path data/processed/Processed_Plant_$(PLANT_ID)_Generation_Data.csv

split-data:
	python spmlops/features.py split-data

train-model:
	python spmlops/modeling/train.py \
		--model-path models/model_plant$(PLANT_ID).pkl

predict:
	python spmlops/modeling/predict.py \
		--features-path data/processed/X_test.csv \
		--model-path models/model_plant$(PLANT_ID).pkl \
		--predictions-path data/processed/predictions_plant$(PLANT_ID).csv

plot:
	python spmlops/plots.py \
		--features-path data/processed/X_test.csv \
		--y-test-path data/processed/y_test.csv \
		--model-path models/model_plant$(PLANT_ID).pkl \
		--predictions-path data/processed/predictions_plant$(PLANT_ID).csv \
		--output-path reports/figures/plot_plant$(PLANT_ID).png
