import numpy as np
import pandas as pd
from scipy.sparse.construct import rand

# -----------------------------------
# 学習データ、テストデータの読み込み
# -----------------------------------
# 学習データ、テストデータの読み込み
train = pd.read_csv('../input/ch01-titanic/train.csv')
test = pd.read_csv('../input/ch01-titanic/test.csv')

# 学習データを特徴量と目的変数に分ける
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

# テストデータは特徴量のみなので、そのままでよい
# コピーはする
test_x = test.copy()


def get_accuracy_wit_cv(n_splits, model, train_x, train_y):
    """CVして精度を求める
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=71)
    # splitによって
    for tr_idx, va_idx in kf.split(train_x):
        print(f'len(tr_idx): {len(tr_idx)}, len(va_idx): {len(va_idx)}')
        # 学習データを学習データとバリデーションデータに分ける
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y)

        # バリデーションデータの予測値を確率で出力する
        va_pred = model.predict_proba(va_x)[:, 1]

        # バリデーションデータでのスコアを計算する
        logloss = log_loss(va_y, va_pred)
        accuracy = accuracy_score(va_y, va_pred > 0.5)

        # そのfoldのスコアを保存する
        scores_logloss.append(logloss)
        scores_accuracy.append(accuracy)
        print("scores_logloss: ", scores_logloss)
        print("scores_accuracy: ", scores_accuracy)
    # 各foldのスコアの平均を出力する
    logloss = np.mean(scores_logloss)
    accuracy = np.mean(scores_accuracy)
    print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')


def set_label_encoding(train_df, test_df, keys):
    """keysで指定したカラムをlabel encodingする。
    train_df, test_dfに適用する
    """
    for key in keys:
        le = LabelEncoder()
        le.fit(train_x[key].fillna('NA'))

        print(f'le.classes_: {le.classes_}')

        train_df[key] = le.transform(train_df[key].fillna('NA'))
        test_df[key] = le.transform(test_df[key].fillna('NA'))


# -----------------------------------
# 特徴量作成
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# 変数PassengerIdを除外する
train_x = train_x.drop(['PassengerId'], axis=1)
test_x = test_x.drop(['PassengerId'], axis=1)

# 変数Name, Ticket, Cabinを除外する
train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

print("-- before label encoding --")
print(train_x)

# それぞれのカテゴリ変数にlabel encodingを適用する
#for c in ['Sex', 'Embarked']:
#    # 学習データに基づいてどう変換するかを定める
#    le = LabelEncoder()
#    le.fit(train_x[c].fillna('NA'))
#
#    # 学習データ、テストデータを変換する
#    train_x[c] = le.transform(train_x[c].fillna('NA'))
#    test_x[c] = le.transform(test_x[c].fillna('NA'))

set_label_encoding(train_x, test_x, ['Sex', 'Embarked'])

print("-- after label encoding --")
print(train_x)


# -----------------------------------
# モデル作成
# -----------------------------------
from xgboost import XGBClassifier

# モデルの作成および学習データを与えての学習
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

# テストデータの予測値を確率で出力する
# クラス1の確率のみを取り出している
# [:, 0]はクラス0の確率
pred = model.predict_proba(test_x)[:, 1]

# テストデータの予測値を二値に変換する
# 2値分類なのでクラス1の確率が0.5未満の場合はクラス0という結論
pred_label = np.where(pred > 0.5, 1, 0)

# 提出用ファイルの作成
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)
# スコア：0.7799（本書中の数値と異なる可能性があります）

# -----------------------------------
# バリデーション
# -----------------------------------
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

# 各foldのスコアを保存するリスト
scores_accuracy = []
scores_logloss = []

# クロスバリデーションを行う
# 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    print("tr_idx:", len(tr_idx), " va_idx:", len(va_idx))
    # 学習データを学習データとバリデーションデータに分ける
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # モデルの学習を行う
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(tr_x, tr_y)

    # バリデーションデータの予測値を確率で出力する
    va_pred = model.predict_proba(va_x)[:, 1]

    # バリデーションデータでのスコアを計算する
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    # そのfoldのスコアを保存する
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)
    print("scores_logloss: ", scores_logloss)
    print("scores_accuracy: ", scores_accuracy)

# 各foldのスコアの平均を出力する
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
# logloss: 0.4270, accuracy: 0.8148（本書中の数値と異なる可能性があります）

# -----------------------------------
# モデルチューニング
# -----------------------------------
import itertools
print("==== model tuning ====")
# チューニング候補とするパラメータを準備する
param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1.0, 2.0, 4.0]
}

# 探索するハイパーパラメータの組み合わせ
param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])
print("param_combinations: ", param_combinations)

# 各パラメータの組み合わせ、それに対するスコアを保存するリスト
params = []
scores = []

# 各パラメータの組み合わせごとに、クロスバリデーションで評価を行う
for max_depth, min_child_weight in param_combinations:

    score_folds = []
    accuracy_folds = []
    # クロスバリデーションを行う
    # 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        # 学習データを学習データとバリデーションデータに分ける
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # モデルの学習を行う
        model = XGBClassifier(n_estimators=20, random_state=71,
                              max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)

        # バリデーションデータでのスコアを計算し、保存する
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        accuracy = accuracy_score(va_y, va_pred > 0.5)
        score_folds.append(logloss)
        accuracy_folds.append(accuracy)

    # 各foldのスコアを平均する
    score_mean = np.mean(score_folds)

    # パラメータの組み合わせ、それに対するスコアを保存する
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

# 最もスコアが良いものをベストなパラメータとする
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'best scores-> accuracy:{accuracy_folds[best_idx]}, loss:{scores[best_idx]}')
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')
# max_depth=7, min_child_weight=2.0のスコアが最もよかった


# -----------------------------------
# ロジスティック回帰用の特徴量の作成
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# 元データをコピーする
train_x2 = train.drop(['Survived'], axis=1)
test_x2 = test.copy()

# 変数PassengerIdを除外する
train_x2 = train_x2.drop(['PassengerId'], axis=1)
test_x2 = test_x2.drop(['PassengerId'], axis=1)

# 変数Name, Ticket, Cabinを除外する
train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# one-hot encodingを行う
cat_cols = ['Sex', 'Embarked', 'Pclass']
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(train_x2[cat_cols].fillna('NA'))

# one-hot encodingのダミー変数の列名を作成する
ohe_columns = []
for i, c in enumerate(cat_cols):
    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]

print("ohe_columns: ", ohe_columns)

# one-hot encodingによる変換を行う
ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)
ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)

# one-hot encoding済みの変数を除外する
train_x2 = train_x2.drop(cat_cols, axis=1)
test_x2 = test_x2.drop(cat_cols, axis=1)

# one-hot encodingで変換された変数を結合する
train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)
test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)

print("one-hot encodingを結合したdataframe")
print(train_x2)

# 数値変数の欠損値を学習データの平均で埋める
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
for col in num_cols:
    train_x2[col].fillna(train_x2[col].mean(), inplace=True)
    test_x2[col].fillna(train_x2[col].mean(), inplace=True)

# 変数Fareを対数変換する
train_x2['Fare'] = np.log1p(train_x2['Fare'])
test_x2['Fare'] = np.log1p(test_x2['Fare'])


# -----------------------------------
# アンサンブル
# -----------------------------------
from sklearn.linear_model import LogisticRegression

# xgboostモデル
model_xgb = XGBClassifier(n_estimators=20, random_state=71)
model_xgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]

# ロジスティック回帰モデル
# xgboostモデルとは異なる特徴量を入れる必要があるので、別途train_x2, test_x2を作成した
model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
model_lr.fit(train_x2, train_y)
pred_lr = model_lr.predict_proba(test_x2)[:, 1]
#pred_label = np.where(pred_lr > 0.5, 1, 0)
# TODO ロジスティック回帰でCVして精度を見てみる
get_accuracy_wit_cv(n_splits=4, model=model_lr, train_x=train_x2, train_y=train_y)

# 予測値の加重平均をとる
pred = pred_xgb * 0.8 + pred_lr * 0.2
pred_label = np.where(pred > 0.5, 1, 0)

# 提出用ファイルの作成
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_second.csv', index=False)
