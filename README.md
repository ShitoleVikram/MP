# Import liabraries
import pandas as pd
import numpy as np
import math
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import pickle

# load datasets
Food_data = pd.read_csv('Diet_Data.csv')
Final_data = pd.read_csv("Final_Data.csv")

# Load Models
MLPmodel = 'F_MLP_model.pkl'
MLP_loaded_mpdel = pickle.load(open(MLPmodel, 'rb'))

Dietmodel = 'Diet_model.pkl'
Diet_loaded_mpdel = pickle.load(open(Dietmodel, 'rb'))

# Pipeline to standerdize data
my_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])
my_pipeline.fit_transform(Final_data)

# Function to calculate BMI


def calculate_bmi(height, weight):
    return weight / (height / 100)**2

# Function to calculate BMR


def calculate_bmr(age, height, weight, gender, activity):
    if gender == 1:
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 655.1 + (9.563 * weight) + (1.850 * height) - (4.676 * age)

    # if activity == 1:
    #     bmr *= 1.2
    # elif activity == 2:
    #     bmr *= 1.375
    # elif activity == 3:
    #     bmr *= 1.55
    # elif activity == 4:
    #     bmr *= 1.725
    # else:
    #     bmr *= 1.9

    return bmr


def activity_val():
    activity = int(input(
        'Enter your activity level (1 sedentary/2 lightly active /3 moderately active /4 very active/5 extremely active): '))

    if activity == 1:
        activity_index = 1.2
    elif activity == 2:
        activity_index = 1.3
    elif activity == 3:
        activity_index = 1.5
    elif activity == 4:
        activity_index = 1.7
    else:
        activity_index = 1.9
    return activity_index


# Take user input
age = int(input("Enter your age: "))
height = float(input("Enter your height in cm: "))
height_m = height/100
weight = float(input("Enter your weight in kg: "))
gender = int(input("Enter your gender (male:1 or female:0): "))
activity = activity_val()
bmi = calculate_bmi(height, weight)
bmr = calculate_bmr(age, height, weight, gender, activity)

scaler = StandardScaler()
inputs_list = []
inputs_list.append([age, weight, height_m, gender, bmi, bmr, activity])
# print(inputs_list)

Data = my_pipeline.transform(inputs_list)
# print(test)

STD_Data = MLP_loaded_mpdel.predict(Data)
caloric_need = STD_Data[0]
caloric_needs = int(caloric_need)


def recommend_foods(BMI, caloric_needs, recommended_foods_df=None):

    if BMI < 18.5:
        prot = BMI*1.5
        carb = BMI*6
        fat = BMI*1.5
    elif 18.5 <= BMI < 24.9:
        prot = BMI*1.2
        carb = BMI*5
        fat = BMI*1
    elif 25 <= BMI < 29.9:
        prot = BMI*1.2
        carb = BMI*4
        fat = BMI*0.8
    else:
        prot = BMI*1.2
        carb = BMI*3
        fat = BMI*0.6

    if recommended_foods_df is None:
        # Create an empty dataframe to store the recommended food items
        recommended_foods_df = pd.DataFrame(
            columns=['Food', 'calories(gm)', 'total_fat(gm)', 'protein(gm)', 'carbohydrate(gm)'])

    food_items = ["Data"]
    total_calories = 0
    total_proteins = 0
    total_fats = 0
    total_carbs = 0
    
    while total_calories < caloric_needs:
        # Calculate remaining caloric need
        remaining_calories = random.randrange(caloric_needs-total_calories)
        remaining_prot = prot - total_proteins
        remaining_carb = carb - total_carbs
        remaining_fat = fat - total_fats

        # Predict food item based on remaining caloric need and user's protein, carbohydrate and fat needs
        nutrient_values = pd.DataFrame({'calories(gm)': [remaining_calories], 'total_fat(gm)': [
                                       remaining_fat], 'protein(gm)': [remaining_prot], 'carbohydrate(gm)': [remaining_carb]})

        food_item = Diet_loaded_mpdel.predict(nutrient_values)[0]

        # Add the food item to the recommendation list
        if food_item not in food_items:
            food_items.append(food_item)

            recommended_food = Food_data[Food_data['Food'] == food_item]
            total_calories += recommended_food['calories(gm)']
            recommended_foods_df = recommended_foods_df.append(
                recommended_food)
            total_proteins += recommended_food['protein(gm)']
            total_fats += recommended_food['total_fat(gm)']
            total_carbs += recommended_food['carbohydrate(gm)']

            # print(f"\n{total_calories}")
            # print(total_proteins)
            # print(total_fats)
            # print(total_carbs)
            # print('________________________________________________________')
            # total_calories += df[df['Food'] == food_item]['calories(gm)'].values[0]
        else:
            total_calories += 10
        # Update the total caloric intake
        # print(total_calories)

    # return food_items
    return recommended_foods_df


recommendations = recommend_foods(bmi, caloric_needs)
print(recommendations)

# print(caloric_needs)
# print(type(caloric_needs))

# Divide caloric needs into breakfast, lunch, and dinner
# calories_breakfast = int(caloric_needs * 0.25)
# calories_lunch = int(caloric_needs * 0.40)
# calories_dinner = int(caloric_needs * 0.35)

# # Filter food dataset by meal type and shuffle the items
# breakfast_items = Food_data[Food_data['Breakfast'] == 1].sample(frac=1).reset_index(drop=True)
# lunch_items = Food_data[Food_data['Lunch'] == 1].sample(frac=1).reset_index(drop=True)
# dinner_items = Food_data[Food_data['Dinner'] == 1].sample(frac=1).reset_index(drop=True)

# # Recommend breakfast
# recommended_breakfast = ''
# for i, row in breakfast_items.iterrows():
#     if row['Calories'] <= calories_breakfast:
#         recommended_breakfast += f"\n{row}\n\n"
#         calories_breakfast -= row['Calories']
#     if calories_breakfast <= 0:
#         break

# # Recommend lunch
# recommended_lunch = ''
# for i, row in lunch_items.iterrows():
#     if row['Calories'] <= calories_lunch:
#         recommended_lunch += f"\n{row}\n\n"
#         calories_lunch -= row['Calories']
#     if calories_lunch <= 0:
#         break

# # Recommend dinner
# recommended_dinner = ''
# for i, row in dinner_items.iterrows():
#     if row['Calories'] <= calories_dinner:
#         recommended_dinner += f"\n{row}\n\n"
#         calories_dinner -= row['Calories']
#     if calories_dinner <= 0:
#         break

# # Print Recommendations
# print(f"Recommended breakfast (total calories: {caloric_needs * 0.25}):{recommended_breakfast}\n")
# print(f"Recommended lunch (total calories: {caloric_needs * 0.40}):{recommended_lunch}\n")
# print(f"Recommended dinner (total calories: {caloric_needs * 0.35}):{recommended_dinner}\n")
