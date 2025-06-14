import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

class ModelTrainer:
    def __init__(self, df, numerical_cols, categorical_cols, binary_cols, target_col='Heart Attack Risk'):
        self.df = df.copy()
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = [col for col in binary_cols if col != target_col]
        self.target_col = target_col
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_prob = None
        self.preprocessor = None

    def _prepare_data(self):
        if self.target_col not in self.df.columns:
            raise ValueError(f'Cột mục tiêu \'{self.target_col}\' không tồn tại trong DataFrame.')
        features = [col for col in self.numerical_cols + self.categorical_cols + self.binary_cols if col in self.df.columns]
        X = self.df[features]
        y = self.df[self.target_col]
        numeric_features = [col for col in self.numerical_cols if col in X.columns]
        categorical_features = [col for col in self.categorical_cols + self.binary_cols if col in X.columns and col not in self.numerical_cols]
        if 'Systolic' in self.df.columns and 'Systolic' not in numeric_features:
            numeric_features.append('Systolic')
        if 'Diastolic' in self.df.columns and 'Diastolic' not in numeric_features:
            numeric_features.append('Diastolic')
        all_features_selected = list(set(numeric_features + categorical_features))
        X = self.df[all_features_selected]
        preprocessor_transformers = []
        if numeric_features:
            preprocessor_transformers.append(('num', StandardScaler(), numeric_features))
        if categorical_features:
            preprocessor_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
        self.preprocessor = ColumnTransformer(
            transformers=preprocessor_transformers,
            remainder='passthrough'
        )
        return X, y

    def train_model(self, test_size=0.2, random_state=42, use_smote=False, smote_sampling_strategy='auto'):
        try:
            X, y = self._prepare_data()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Tạo pipeline chỉ với preprocessor và classifier
            steps = []
            steps.append(('preprocessor', self.preprocessor))
            steps.append(('classifier', LogisticRegression(random_state=random_state, solver='liblinear', max_iter=1000)))
            self.model = Pipeline(steps=steps)
            if use_smote:
                X_train_transformed = self.model.named_steps['preprocessor'].fit_transform(self.X_train)
                smote = SMOTE(sampling_strategy=smote_sampling_strategy, random_state=random_state)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, self.y_train)
                self.model.named_steps['classifier'].fit(X_train_resampled, y_train_resampled)
            else:
                self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            self.y_prob = self.model.predict_proba(self.X_test)[:, 1] 
            metrics = self.evaluate_model()
            return metrics
        except Exception as e:
            raise Exception(f'Lỗi khi huấn luyện mô hình: {e}')

    def evaluate_model(self):
        if self.model is None or self.y_test is None or self.y_pred is None:
            raise Exception('Mô hình chưa được huấn luyện hoặc dữ liệu kiểm tra chưa có.')
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        roc_auc = roc_auc_score(self.y_test, self.y_prob)
        cm = confusion_matrix(self.y_test, self.y_pred)
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
        return metrics

    def plot_roc_curve(self):
        if self.y_test is None or self.y_prob is None:
            return None
        
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='orange', label=f'ROC curve (area = {self.evaluate_model()['roc_auc']:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def plot_confusion_matrix(self):
        if self.y_test is None or self.y_pred is None:
            return None
            
        cm = confusion_matrix(self.y_test, self.y_pred)
        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Dự đoán Âm', 'Dự đoán Dương'],
                    yticklabels=['Thực tế Âm', 'Thực tế Dương'])
        plt.title('Ma trận nhầm lẫn')
        plt.xlabel('Dự đoán')
        plt.ylabel('Thực tế')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def get_feature_names_out(self):
        if self.preprocessor is None:
            return []
        try:
            feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        except AttributeError as e:
            print(f'Lỗi khi lấy tên đặc trưng: {e}. Sử dụng phương pháp thay thế.')
            feature_names = []
            for name, transformer, cols in self.preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(cols)
                elif name == 'cat':
                    ohe_feature_names = transformer.get_feature_names_out(cols)
                    feature_names.extend(ohe_feature_names)
                elif name == 'remainder':
                    pass
        return feature_names

    def get_model_coefficients(self):
        if self.model is None:
            return None
        classifier = self.model.named_steps['classifier']
        if hasattr(classifier, 'coef_') and len(classifier.coef_[0]) > 0:
            coefs = classifier.coef_[0]
            feature_names = self.get_feature_names_out()
            
            if len(coefs) == len(feature_names):
                return pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
            else:
                print(f'Warning: Số lượng hệ số ({len(coefs)}) không khớp số lượng đặc trưng ({len(feature_names)}). Generating placeholder feature names.')
                placeholder_features = [f'Feature_{i+1}' for i in range(len(coefs))]
                return pd.DataFrame({'Feature': placeholder_features, 'Coefficient': coefs})
        return None