import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import threading
import queue
import os

class HeartRiskAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phân tích nguy cơ đau tim")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Biến lưu trữ dữ liệu
        self.df = None
        self.model = None
        self.features = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        
        # Tạo giao diện
        self.create_widgets()
        
        # Queue để giao tiếp giữa thread và giao diện
        self.queue = queue.Queue()
        self.update_gui()
    
    def create_widgets(self):
        # Frame chính
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Notebook (tab control)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Tải dữ liệu
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Tải dữ liệu")
        
        # Tab 2: Tiền xử lý
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Tiền xử lý")
        
        # Tab 3: Mô hình
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Mô hình")
        
        # Tab 4: Phân tích
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="Phân tích")
        
        # Tab 5: Tương quan
        self.tab5 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab5, text="Tương quan")
        
        # Thiết lập nội dung cho từng tab
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()
        self.setup_tab5()
        
        # Thanh trạng thái
        self.status_var = tk.StringVar()
        self.status_var.set("Sẵn sàng")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
    
    def setup_tab1(self):
        # Frame cho tải dữ liệu
        frame = ttk.LabelFrame(self.tab1, text="Tải dữ liệu", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Nút chọn file
        ttk.Button(frame, text="Chọn file CSV", command=self.load_data).pack(pady=10)
        
        # Hiển thị thông tin dữ liệu
        self.data_info = tk.Text(frame, height=15, width=80)
        self.data_info.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Frame hiển thị dữ liệu
        data_frame = ttk.LabelFrame(self.tab1, text="Xem dữ liệu", padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview để hiển thị dữ liệu
        self.tree = ttk.Treeview(data_frame)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar cho Treeview
        scrollbar = ttk.Scrollbar(data_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
    
    def setup_tab2(self):
        # Frame cho tiền xử lý
        frame = ttk.LabelFrame(self.tab2, text="Tiền xử lý dữ liệu", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Nút tiền xử lý
        ttk.Button(frame, text="Tiền xử lý dữ liệu", command=self.preprocess_data).pack(pady=10)
        
        # Hiển thị thông tin tiền xử lý
        self.preprocess_info = tk.Text(frame, height=15, width=80)
        self.preprocess_info.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Frame hiển thị dữ liệu sau tiền xử lý
        data_frame = ttk.LabelFrame(self.tab2, text="Dữ liệu sau tiền xử lý", padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview để hiển thị dữ liệu
        self.tree_processed = ttk.Treeview(data_frame)
        self.tree_processed.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar cho Treeview
        scrollbar = ttk.Scrollbar(data_frame, orient="vertical", command=self.tree_processed.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_processed.configure(yscrollcommand=scrollbar.set)
    
    def setup_tab3(self):
        # Frame cho mô hình
        frame = ttk.LabelFrame(self.tab3, text="Huấn luyện mô hình", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Nút huấn luyện
        ttk.Button(frame, text="Huấn luyện mô hình", command=self.train_model).pack(pady=10)
        
        # Hiển thị thông tin mô hình
        self.model_info = tk.Text(frame, height=20, width=80)
        self.model_info.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Frame cho biểu đồ
        plot_frame = ttk.LabelFrame(self.tab3, text="Đánh giá mô hình", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas cho biểu đồ
        self.model_canvas_frame = ttk.Frame(plot_frame)
        self.model_canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    def setup_tab4(self):
        # Frame cho phân tích
        frame = ttk.LabelFrame(self.tab4, text="Phân tích mô hình", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Nút phân tích
        ttk.Button(frame, text="Phân tích mô hình", command=self.analyze_model).pack(pady=10)
        
        # Hiển thị thông tin phân tích
        self.analysis_info = tk.Text(frame, height=10, width=80)
        self.analysis_info.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Frame cho biểu đồ
        plot_frame = ttk.LabelFrame(self.tab4, text="Biểu đồ phân tích", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas cho biểu đồ
        self.analysis_canvas_frame = ttk.Frame(plot_frame)
        self.analysis_canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    def setup_tab5(self):
        # Frame cho tương quan
        frame = ttk.LabelFrame(self.tab5, text="Phân tích tương quan", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Nút phân tích tương quan
        ttk.Button(frame, text="Phân tích tương quan", command=self.analyze_correlations).pack(pady=10)
        
        # Combobox để chọn biểu đồ
        ttk.Label(frame, text="Chọn biểu đồ:").pack(pady=5)
        self.corr_plot_var = tk.StringVar()
        corr_plots = ["Ma trận tương quan", "Biểu đồ cặp", "Tương quan với biến mục tiêu", 
                      "Phân cụm tương quan", "Biến phân loại (1-6)", "Biến phân loại (7-11)"]
        self.corr_combo = ttk.Combobox(frame, textvariable=self.corr_plot_var, values=corr_plots)
        self.corr_combo.pack(pady=5)
        self.corr_combo.current(0)
        self.corr_combo.bind("<<ComboboxSelected>>", self.show_correlation_plot)
        
        # Frame cho biểu đồ
        plot_frame = ttk.LabelFrame(self.tab5, text="Biểu đồ tương quan", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas cho biểu đồ
        self.corr_canvas_frame = ttk.Frame(plot_frame)
        self.corr_canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        
        self.status_var.set("Đang tải dữ liệu...")
        self.progress_var.set(10)
        
        # Sử dụng thread để không làm đơ giao diện
        threading.Thread(target=self._load_data_thread, args=(file_path,)).start()
    
    def _load_data_thread(self, file_path):
        try:
            # Tải dữ liệu
            self.df = pd.read_csv(file_path)
            
            # Chuẩn bị thông tin để hiển thị
            info_text = f"Shape: {self.df.shape}\n\n"
            info_text += f"Columns: {self.df.columns.tolist()}\n\n"
            info_text += "Missing values:\n"
            info_text += str(self.df.isnull().sum()) + "\n\n"
            info_text += "Class distribution:\n"
            info_text += str(self.df['Heart Attack Risk'].value_counts(normalize=True))
            
            # Chuẩn bị dữ liệu cho Treeview
            self.queue.put(("update_data_info", info_text))
            self.queue.put(("update_treeview", self.df))
            self.queue.put(("update_status", "Dữ liệu đã được tải thành công"))
            self.queue.put(("update_progress", 100))
            
        except Exception as e:
            self.queue.put(("show_error", f"Lỗi khi tải dữ liệu: {str(e)}"))
            self.queue.put(("update_status", "Lỗi khi tải dữ liệu"))
            self.queue.put(("update_progress", 0))
    
    def update_treeview(self, df, treeview):
        # Xóa dữ liệu cũ
        for item in treeview.get_children():
            treeview.delete(item)
        
        # Cập nhật cột
        columns = list(df.columns)
        treeview["columns"] = columns
        treeview["show"] = "headings"
        
        for col in columns:
            treeview.heading(col, text=col)
            treeview.column(col, width=100)
        
        # Thêm dữ liệu
        for i, row in df.head(100).iterrows():
            treeview.insert("", "end", values=list(row))
    
    def preprocess_data(self):
        if self.df is None:
            messagebox.showerror("Lỗi", "Vui lòng tải dữ liệu trước")
            return
        
        self.status_var.set("Đang tiền xử lý dữ liệu...")
        self.progress_var.set(10)
        
        # Sử dụng thread để không làm đơ giao diện
        threading.Thread(target=self._preprocess_data_thread).start()
    
    def _preprocess_data_thread(self):
        try:
            # Xử lý huyết áp
            self.df[['Systolic_BP', 'Diastolic_BP']] = self.df['Blood Pressure'].str.split('/', expand=True).astype(float)
            self.df.drop('Blood Pressure', axis=1, inplace=True)
            
            # Xác định các loại đặc trưng
            self.binary_cols = ['Diabetes', 'Family History', 'Smoking', 'Obesity', 
                          'Alcohol Consumption', 'Previous Heart Problems', 'Medication Use']
            self.categorical_cols = ['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere']
            self.numerical_cols = ['Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week',
                             'Stress Level', 'Sedentary Hours Per Day', 'Income', 'BMI',
                             'Triglycerides', 'Physical Activity Days Per Week', 
                             'Sleep Hours Per Day', 'Systolic_BP', 'Diastolic_BP']
            
            # Xử lý giá trị thiếu
            for col in self.numerical_cols:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            
            for col in self.categorical_cols + self.binary_cols:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            
            # Xử lý ngoại lai
            for col in self.numerical_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, 
                                  np.where(self.df[col] < lower_bound, lower_bound, self.df[col]))
            
            # Chuẩn bị thông tin để hiển thị
            info_text = "Tiền xử lý dữ liệu hoàn tất:\n\n"
            info_text += f"Số lượng biến số: {len(self.numerical_cols)}\n"
            info_text += f"Số lượng biến phân loại: {len(self.categorical_cols)}\n"
            info_text += f"Số lượng biến nhị phân: {len(self.binary_cols)}\n\n"
            info_text += "Các biến số:\n" + str(self.numerical_cols) + "\n\n"
            info_text += "Các biến phân loại:\n" + str(self.categorical_cols) + "\n\n"
            info_text += "Các biến nhị phân:\n" + str(self.binary_cols)
            
            self.queue.put(("update_preprocess_info", info_text))
            self.queue.put(("update_processed_treeview", self.df))
            self.queue.put(("update_status", "Tiền xử lý dữ liệu hoàn tất"))
            self.queue.put(("update_progress", 100))
            
        except Exception as e:
            self.queue.put(("show_error", f"Lỗi khi tiền xử lý dữ liệu: {str(e)}"))
            self.queue.put(("update_status", "Lỗi khi tiền xử lý dữ liệu"))
            self.queue.put(("update_progress", 0))
    
    def train_model(self):
        if self.df is None or not hasattr(self, 'numerical_cols'):
            messagebox.showerror("Lỗi", "Vui lòng tiền xử lý dữ liệu trước")
            return
        
        self.status_var.set("Đang huấn luyện mô hình...")
        self.progress_var.set(10)
        
        # Sử dụng thread để không làm đơ giao diện
        threading.Thread(target=self._train_model_thread).start()
    
    def _train_model_thread(self):
        try:
            # Chuẩn bị dữ liệu
            # Tiền xử lý cột
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.numerical_cols),
                    ('cat', OneHotEncoder(drop='first'), self.categorical_cols)
                ],
                remainder='passthrough'
            )
            
            # Tách đặc trưng và mục tiêu
            X = self.df.drop(['Patient ID', 'Heart Attack Risk'], axis=1)
            y = self.df['Heart Attack Risk']
            
            # Áp dụng tiền xử lý
            X_processed = preprocessor.fit_transform(X)
            
            # Lấy tên đặc trưng sau mã hóa
            cat_encoder = preprocessor.named_transformers_['cat']
            cat_features = cat_encoder.get_feature_names_out(self.categorical_cols)
            self.features = np.concatenate([self.numerical_cols, cat_features, self.binary_cols])
            
            # Xử lý mất cân bằng lớp
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_processed, y)
            
            # Chia dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
            )
            
            # Huấn luyện mô hình
            self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            self.model.fit(X_train, y_train)
            
            # Dự đoán
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Đánh giá mô hình
            report = classification_report(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            # Kiểm định chéo
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_model = LogisticRegression(max_iter=1000, class_weight='balanced')
            
            # Đánh giá các chỉ số
            cv_scores = {}
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            for metric in scoring:
                scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring=metric)
                cv_scores[metric] = (np.mean(scores), np.std(scores))
            
            # Chuẩn bị thông tin để hiển thị
            info_text = "Kết quả huấn luyện mô hình:\n\n"
            info_text += f"Số lượng mẫu huấn luyện: {X_train.shape[0]}\n"
            info_text += f"Số lượng mẫu kiểm tra: {X_test.shape[0]}\n\n"
            info_text += "Classification Report:\n"
            info_text += report + "\n\n"
            info_text += f"ROC-AUC Score: {roc_auc:.4f}\n\n"
            info_text += "Kết quả kiểm định chéo (5-fold):\n"
            for metric, (mean, std) in cv_scores.items():
                info_text += f"{metric}: {mean:.4f} ± {std:.4f}\n"
            
            # Vẽ ma trận nhầm lẫn
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Dự đoán')
            ax.set_ylabel('Thực tế')
            ax.set_title('Ma trận nhầm lẫn')
            
            # Hiển thị biểu đồ trong giao diện
            self.queue.put(("update_model_info", info_text))
            self.queue.put(("update_model_plot", fig))
            self.queue.put(("update_status", "Huấn luyện mô hình hoàn tất"))
            self.queue.put(("update_progress", 100))
            
            # Lưu thông tin cho phân tích
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.y_pred = y_pred
            self.y_proba = y_proba
            self.preprocessor = preprocessor
            
        except Exception as e:
            self.queue.put(("show_error", f"Lỗi khi huấn luyện mô hình: {str(e)}"))
            self.queue.put(("update_status", "Lỗi khi huấn luyện mô hình"))
            self.queue.put(("update_progress", 0))
    
    def analyze_model(self):
        if self.model is None:
            messagebox.showerror("Lỗi", "Vui lòng huấn luyện mô hình trước")
            return
        
        self.status_var.set("Đang phân tích mô hình...")
        self.progress_var.set(10)
        
        # Sử dụng thread để không làm đơ giao diện
        threading.Thread(target=self._analyze_model_thread).start()
    
    def _analyze_model_thread(self):
        try:
            # Phân tích hệ số và tỷ lệ odds
            coefficients = pd.DataFrame({
                'Feature': self.features,
                'Coefficient': self.model.coef_[0],
                'Odds_Ratio': np.exp(self.model.coef_[0])
            })
            
            # Sắp xếp theo độ lớn
            coefficients = coefficients.sort_values(by='Odds_Ratio', ascending=False)
            
            # Chuẩn bị thông tin để hiển thị
            info_text = "Phân tích hệ số mô hình:\n\n"
            info_text += "Top 10 đặc trưng tăng nguy cơ đau tim:\n"
            info_text += str(coefficients.head(10)[['Feature', 'Odds_Ratio']]) + "\n\n"
            info_text += "Top 10 đặc trưng giảm nguy cơ đau tim:\n"
            info_text += str(coefficients.tail(10)[['Feature', 'Odds_Ratio']])
            
            # Vẽ biểu đồ tỷ lệ odds
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = pd.concat([coefficients.head(5), coefficients.tail(5)])
            sns.barplot(data=top_features, x='Odds_Ratio', y='Feature', hue='Feature', legend=False, palette='viridis', ax=ax)
            ax.axvline(x=1, color='r', linestyle='--')
            ax.set_title('Các đặc trưng ảnh hưởng đến nguy cơ đau tim (Tỷ lệ Odds)')
            
            # Hiển thị biểu đồ trong giao diện
            self.queue.put(("update_analysis_info", info_text))
            self.queue.put(("update_analysis_plot", fig))
            self.queue.put(("update_status", "Phân tích mô hình hoàn tất"))
            self.queue.put(("update_progress", 100))
            
        except Exception as e:
            self.queue.put(("show_error", f"Lỗi khi phân tích mô hình: {str(e)}"))
            self.queue.put(("update_status", "Lỗi khi phân tích mô hình"))
            self.queue.put(("update_progress", 0))
    
    def analyze_correlations(self):
        if self.df is None:
            messagebox.showerror("Lỗi", "Vui lòng tải dữ liệu trước")
            return
        
        self.status_var.set("Đang phân tích tương quan...")
        self.progress_var.set(10)
        
        # Sử dụng thread để không làm đơ giao diện
        threading.Thread(target=self._analyze_correlations_thread).start()
    
    def _analyze_correlations_thread(self):
        try:
            # Tính ma trận tương quan
            self.corr_matrix = self.df.select_dtypes(include=['float64', 'int64']).corr()
            
            # Vẽ biểu đồ tương quan đầu tiên (mặc định)
            self.show_correlation_plot(None)
            
            self.queue.put(("update_status", "Phân tích tương quan hoàn tất"))
            self.queue.put(("update_progress", 100))
            
        except Exception as e:
            self.queue.put(("show_error", f"Lỗi khi phân tích tương quan: {str(e)}"))
            self.queue.put(("update_status", "Lỗi khi phân tích tương quan"))
            self.queue.put(("update_progress", 0))
    
    def show_correlation_plot(self, event):
        if not hasattr(self, 'corr_matrix'):
            return
        
        plot_type = self.corr_plot_var.get()
        
        try:
            plt.close('all')  # Đóng tất cả các biểu đồ cũ
            
            if plot_type == "Ma trận tương quan":
                fig, ax = plt.subplots(figsize=(12, 10))
                mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool))
                sns.heatmap(self.corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                            center=0, linewidths=0.5, ax=ax)
                ax.set_title('Ma trận tương quan giữa các biến số')
                
            elif plot_type == "Biểu đồ cặp":
                important_vars = ['Age', 'Cholesterol', 'Heart Rate', 'Systolic_BP', 
                                 'Diastolic_BP', 'BMI', 'Heart Attack Risk']
                fig = sns.pairplot(self.df[important_vars], hue='Heart Attack Risk', palette='viridis')
                fig.fig.suptitle('Biểu đồ cặp giữa các biến quan trọng', y=1.02)
                
            elif plot_type == "Tương quan với biến mục tiêu":
                fig, ax = plt.subplots(figsize=(10, 8))
                target_corr = self.corr_matrix['Heart Attack Risk'].sort_values(ascending=False)
                target_df = pd.DataFrame({
                    'Correlation': target_corr.values[1:11],
                    'Feature': target_corr.index[1:11]
                })
                sns.barplot(data=target_df, x='Correlation', y='Feature', hue='Feature', legend=False, palette='viridis', ax=ax)
                ax.set_title('Top 10 biến có tương quan cao nhất với nguy cơ đau tim')
                
            elif plot_type == "Phân cụm tương quan":
                fig = plt.figure(figsize=(12, 10))
                g = sns.clustermap(self.corr_matrix, cmap='coolwarm', center=0, 
                              linewidths=0.5, figsize=(12, 10), 
                              dendrogram_ratio=(.1, .2))
                plt.title('Phân cụm tương quan giữa các biến', y=1.02)
                fig = g.fig
                
            elif plot_type == "Biến phân loại (1-6)":
                categorical_cols = ['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere', 'Diabetes']
                fig, axes = plt.subplots(2, 3, figsize=(14, 10))
                axes = axes.flatten()
                
                for i, col in enumerate(categorical_cols):
                    sns.countplot(x=col, hue='Heart Attack Risk', data=self.df, palette='viridis', ax=axes[i])
                    axes[i].set_title(f'Phân bố {col} theo nguy cơ đau tim')
                    axes[i].tick_params(axis='x', rotation=45)
                
                fig.tight_layout()
                
            elif plot_type == "Biến phân loại (7-11)":
                categorical_cols = ['Family History', 'Smoking', 'Obesity', 
                                   'Alcohol Consumption', 'Previous Heart Problems']
                fig, axes = plt.subplots(2, 3, figsize=(14, 10))
                axes = axes.flatten()
                
                for i, col in enumerate(categorical_cols):
                    if i < len(axes):
                        sns.countplot(x=col, hue='Heart Attack Risk', data=self.df, palette='viridis', ax=axes[i])
                        axes[i].set_title(f'Phân bố {col} theo nguy cơ đau tim')
                        axes[i].tick_params(axis='x', rotation=45)
                
                # Ẩn các trục thừa
                for j in range(len(categorical_cols), len(axes)):
                    axes[j].set_visible(False)
                    
                fig.tight_layout()
            
            # Hiển thị biểu đồ trong giao diện
            self.queue.put(("update_correlation_plot", fig))
            
        except Exception as e:
            self.queue.put(("show_error", f"Lỗi khi vẽ biểu đồ tương quan: {str(e)}"))
    
    def update_gui(self):
        """Cập nhật giao diện từ queue"""
        try:
            while not self.queue.empty():
                action, data = self.queue.get()
                
                if action == "update_data_info":
                    self.data_info.delete(1.0, tk.END)
                    self.data_info.insert(tk.END, data)
                
                elif action == "update_treeview":
                    self.update_treeview(data, self.tree)
                
                elif action == "update_preprocess_info":
                    self.preprocess_info.delete(1.0, tk.END)
                    self.preprocess_info.insert(tk.END, data)
                
                elif action == "update_processed_treeview":
                    self.update_treeview(data, self.tree_processed)
                
                elif action == "update_model_info":
                    self.model_info.delete(1.0, tk.END)
                    self.model_info.insert(tk.END, data)
                
                elif action == "update_model_plot":
                    for widget in self.model_canvas_frame.winfo_children():
                        widget.destroy()
                    
                    canvas = FigureCanvasTkAgg(data, self.model_canvas_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                elif action == "update_analysis_info":
                    self.analysis_info.delete(1.0, tk.END)
                    self.analysis_info.insert(tk.END, data)
                
                elif action == "update_analysis_plot":
                    for widget in self.analysis_canvas_frame.winfo_children():
                        widget.destroy()
                    
                    canvas = FigureCanvasTkAgg(data, self.analysis_canvas_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                elif action == "update_correlation_plot":
                    for widget in self.corr_canvas_frame.winfo_children():
                        widget.destroy()
                    
                    canvas = FigureCanvasTkAgg(data, self.corr_canvas_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                elif action == "update_status":
                    self.status_var.set(data)
                
                elif action == "update_progress":
                    self.progress_var.set(data)
                
                elif action == "show_error":
                    messagebox.showerror("Lỗi", data)
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi cập nhật giao diện: {str(e)}")
        
        # Lập lịch kiểm tra queue sau 100ms
        self.root.after(100, self.update_gui)

# Hàm chính để chạy ứng dụng
def main():
    root = tk.Tk()
    app = HeartRiskAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
