import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Оптимизация функции добавления "магических" фичей
def add_magic_feature(df, df_test):
    combined = pd.concat([df, df_test])
    magic_features = combined.apply(lambda col: col.map(col.value_counts()), axis=0)
    df_magic = magic_features.iloc[:len(df)].reset_index(drop=True)
    df_test_magic = magic_features.iloc[len(df):].reset_index(drop=True)
    
    df = pd.concat([df.reset_index(drop=True), df_magic.add_suffix('_magic')], axis=1)
    df_test = pd.concat([df_test.reset_index(drop=True), df_test_magic.add_suffix('_magic')], axis=1)
    
    return df, df_test

# Основной блок, защищённый проверкой
if __name__ == "__main__":
    logging.info('Start of the process...')
    
    # Загрузка данных
    logging.info('Loading datasets...')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')

    # Определение признаков и целевой переменной
    X = train.drop(['ID_code', 'target'], axis=1)
    y = train['target']
    X_test = test.drop(['ID_code'], axis=1)

    # Добавление магических фичей
    logging.info('Adding magic features...')
    X, X_test = add_magic_feature(X, X_test)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    # Использование StratifiedKFold для более точного разделения данных
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Переменные для отслеживания лучшего результата
    best_auc = 0
    best_y_pred = None

    # Цикл по фолдам
    for fold, (train_index, val_index) in enumerate(skf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Инициализация модели LightGBM с регуляризацией
        lgb_model = lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            n_estimators=5000,
            learning_rate=0.04,
            num_leaves=4,
            max_depth=-1,
            class_weight='balanced',
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
        
        # Обучение модели с использованием ранней остановки
        logging.info(f'Training LightGBM model for fold {fold + 1}...')
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(50)
            ]
        )
        
        # Предсказания вероятностей на валидационном наборе данных
        y_val_pred = lgb_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred)
        logging.info(f'Validation AUC-ROC for fold {fold + 1}: {val_auc}')

        # Если текущий фолд показывает лучший результат, сохранить его предсказания на тестовом наборе
        if val_auc > best_auc:
            best_auc = val_auc
            best_y_pred = lgb_model.predict_proba(X_test_scaled)[:, 1]
            logging.info(f'New best AUC-ROC found: {best_auc} on fold {fold + 1}')

    # Формирование финального файла для отправки на Kaggle
    logging.info('Generating submission file with the best model...')
    submission = pd.DataFrame({
        'ID_code': test['ID_code'],
        'target': best_y_pred
    })
    submission.to_csv('submission.csv', index=False)

    logging.info(f'Best Validation AUC-ROC across all folds: {best_auc}')
    logging.info('End of the process.')