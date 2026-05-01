import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Считываем файл
data = pd.read_excel('metan.xlsx')

# Выделяем X и Y
X = data.iloc[:, :-3]
y = data.iloc[:, -2]

# Шаг 2: Построим матрицу корреляции
correlation_matrix = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Настраиваем RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=10)

# Функция для оценки моделей
def evaluate_model(model, X, y):
    mae_cv_scores = -cross_val_score(model, X, y, cv=rkf, scoring='neg_mean_absolute_error')
    r2_cv_scores = cross_val_score(model, X, y, cv=rkf, scoring='r2')
    mae_cv = mae_cv_scores.mean()
    mae_cv_std = mae_cv_scores.std()
    r2_cv = r2_cv_scores.mean()
    r2_cv_std = r2_cv_scores.std()
    return mae_cv, mae_cv_std, r2_cv, r2_cv_std

# Только модель случайного леса
models = {
    "Random Forest": RandomForestRegressor()
}

# Оценка модели
results_initial = {}
for name, model in models.items():
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    mae_cv, mae_cv_std, r2_cv, r2_cv_std = evaluate_model(pipeline, X_train, y_train)
    results_initial[name] = {'MAE_CV': mae_cv, 'MAE_CV_STD': mae_cv_std, 'R2_CV': r2_cv, 'R2_CV_STD': r2_cv_std}
    print(f"Initial {name}: CV MAE = {mae_cv:.2f} ± {mae_cv_std:.2f}, CV R2 = {r2_cv:.2f} ± {r2_cv_std:.2f}")

# Оптимизация для случайного леса
param_grid_rf = {
    'model__max_depth': [5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, None],
    'model__n_estimators': [50, 100, 150, 200, 300, 400]
}
pipeline_rf = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor())])
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=rkf, scoring='r2')
grid_search_rf.fit(X_train, y_train)

print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")
print(f"Best CV R2 score for Random Forest: {grid_search_rf.best_score_}")

# Повторная оценка модели с оптимизированными гиперпараметрами
best_rf = grid_search_rf.best_estimator_
mae_cv_optimized, mae_cv_std_optimized, r2_cv_optimized, r2_cv_std_optimized = evaluate_model(best_rf, X_train, y_train)
print(f"Optimized Random Forest: CV MAE = {mae_cv_optimized:.2f} ± {mae_cv_std_optimized:.2f}, CV R2 = {r2_cv_optimized:.2f} ± {r2_cv_std_optimized:.2f}")

# Построение графиков для лучших моделей
best_rf.fit(X_train, y_train)
y_pred_train = best_rf.predict(X_train)
y_pred_test = best_rf.predict(X_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_train, alpha=0.7, color='g', label='Train Data')
plt.scatter(y_test, y_pred_test, alpha=0.7, color='b', label='Test Data')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Model')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Optimized Random Forest Predictions')
plt.legend()
plt.text(0.05, 0.95, f'Initial CV MAE = {results_initial["Random Forest"]["MAE_CV"]:.2f} ± {results_initial["Random Forest"]["MAE_CV_STD"]:.2f}, '
                    f'CV R2 = {results_initial["Random Forest"]["R2_CV"]:.2f} ± {results_initial["Random Forest"]["R2_CV_STD"]:.2f}\n'
                    f'Optimized CV MAE = {mae_cv_optimized:.2f} ± {mae_cv_std_optimized:.2f}, '
                    f'CV R2 = {r2_cv_optimized:.2f} ± {r2_cv_std_optimized:.2f}\n'
                    f'Train MAE = {mae_train:.2f}, R2 = {r2_train:.2f}\n'
                    f'Test MAE = {mae_test:.2f}, R2 = {r2_test:.2f}', 
                    transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.show()

# Сохраняем результаты в Excel
results_df = pd.DataFrame(results_initial).T
optimized_results_df = pd.DataFrame({'Optimized_MAE_CV': [mae_cv_optimized], 
                                     'Optimized_MAE_CV_STD': [mae_cv_std_optimized], 
                                     'Optimized_R2_CV': [r2_cv_optimized], 
                                     'Optimized_R2_CV_STD': [r2_cv_std_optimized]})

# Создаем отдельные DataFrames для истинных и предсказанных значений
test_results_df = pd.DataFrame({'True_Values_Test': y_test, 'Predicted_Values_Test': y_pred_test})
train_results_df = pd.DataFrame({'True_Values_Train': y_train, 'Predicted_Values_Train': y_pred_train})

# Пишем результаты в Excel
with pd.ExcelWriter('model_evaluation_results_with_predictions.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Model_Evaluation')
    optimized_results_df.to_excel(writer, sheet_name='Optimized_Evaluation')
    test_results_df.to_excel(writer, sheet_name='Test_Predictions')
    train_results_df.to_excel(writer, sheet_name='Train_Predictions')

# Считываем экспериментальные данные
exp_data = pd.read_excel('exp1.xlsx')

# Выделяем X_exp и y_exp
X_exp = exp_data.iloc[:, :-3]
y_exp = exp_data.iloc[:, -2]

# Предсказания для экспериментальных данных
y_pred_exp = best_rf.predict(X_exp)

# Оценка модели для экспериментальных данных
mae_exp = mean_absolute_error(y_exp, y_pred_exp)
r2_exp = r2_score(y_exp, y_pred_exp)

# Построение графика с экспериментальными данными
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_train, alpha=0.7, color='g', label='Train Data')
plt.scatter(y_test, y_pred_test, alpha=0.7, color='b', label='Test Data')
plt.scatter(y_exp, y_pred_exp, alpha=0.7, color='r', label='Experimental Data')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Model')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Optimized Random Forest Predictions with Experimental Data')
plt.legend()
plt.text(0.05, 0.95, f'Initial CV MAE = {results_initial["Random Forest"]["MAE_CV"]:.2f} ± {results_initial["Random Forest"]["MAE_CV_STD"]:.2f}, '
                    f'CV R2 = {results_initial["Random Forest"]["R2_CV"]:.2f} ± {results_initial["Random Forest"]["R2_CV_STD"]:.2f}\n'
                    f'Optimized CV MAE = {mae_cv_optimized:.2f} ± {mae_cv_std_optimized:.2f}, '
                    f'CV R2 = {r2_cv_optimized:.2f} ± {r2_cv_std_optimized:.2f}\n'
                    f'Train MAE = {mae_train:.2f}, R2 = {r2_train:.2f}\n'
                    f'Test MAE = {mae_test:.2f}, R2 = {r2_test:.2f}\n'
                    f'Exp MAE = {mae_exp:.2f}, R2 = {r2_exp:.2f}', 
                    transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.show()

# Создаем отдельный DataFrame для истинных и предсказанных значений эксперимента
exp_results_df = pd.DataFrame({'True_Values_Exp': y_exp, 'Predicted_Values_Exp': y_pred_exp})

# Пишем результаты в Excel
with pd.ExcelWriter('model_evaluation_results_with_predictions.xlsx', mode='a', engine='openpyxl') as writer:
    exp_results_df.to_excel(writer, sheet_name='Exp_Predictions')
print('end')
#Изменение, добавь 
#новая ветка
