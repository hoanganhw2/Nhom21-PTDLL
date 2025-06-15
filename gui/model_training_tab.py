import tkinter as tk
from tkinter import ttk, messagebox
import threading
from PIL import Image, ImageTk
import io
import base64

from core.model_trainer import ModelTrainer

class ModelTrainingTab:
    def __init__(self, master, app_instance):
        self.master = master
        self.app = app_instance
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        self.model_trainer = None
        self.frame = ttk.Frame(self.master, padding='15')
        self.frame.pack(expand=True, fill='both')
        config_frame = ttk.LabelFrame(self.frame, text='Cấu hình Huấn luyện Mô hình', padding='15')
        config_frame.pack(pady=10, padx=10, fill='x')
        ttk.Label(config_frame, text='Tỷ lệ Tập kiểm tra:').grid(row=0, column=0, sticky='w', pady=5)
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Scale(config_frame, from_=0.1, to=0.5,
                  variable=self.test_size_var, orient='horizontal', length=200).grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Label(config_frame, textvariable=self.test_size_var).grid(row=0, column=2, sticky='w')
        ttk.Label(config_frame, text='Sử dụng SMOTE để cân bằng dữ liệu:').grid(row=1, column=0, sticky='w', pady=5)
        self.use_smote_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text='Bật SMOTE', variable=self.use_smote_var).grid(row=1, column=1, sticky='w')
        self.train_button = ttk.Button(config_frame, text='Huấn luyện Mô hình', command=self._start_model_training)
        self.train_button.grid(row=2, column=0, columnspan=3, pady=15)
        result_frame = ttk.LabelFrame(self.frame, text='Kết quả Đánh giá Mô hình', padding='15')
        result_frame.pack(pady=10, padx=10, fill='both', expand=True)
        self.metrics_label = ttk.Label(result_frame, text='Các chỉ số:\n', justify=tk.LEFT, font=('Arial', 10))
        self.metrics_label.pack(pady=5, anchor='w')
        plot_frame = ttk.Frame(result_frame)
        plot_frame.pack(pady=10, fill='both', expand=True)
        self.cm_image_label = ttk.Label(plot_frame)
        self.cm_image_label.pack(side=tk.LEFT, padx=10, expand=True, fill='both')
        self.roc_image_label = ttk.Label(plot_frame)
        self.roc_image_label.pack(side=tk.RIGHT, padx=10, expand=True, fill='both')
        self.status_label = ttk.Label(self.frame, text='', foreground='blue')
        self.status_label.pack(pady=10)
        self._disable_buttons()

    def update_dataframe_and_cols(self, df, numerical_cols, categorical_cols, binary_cols):
        self.df = df
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols
        self.status_label.config(text='Dữ liệu và phân loại biến đã sẵn sàng để huấn luyện.')
        self._enable_buttons()
        
    def _disable_buttons(self):
        self.train_button.config(state='disabled')

    def _enable_buttons(self):
        if self.df is not None:
            self.train_button.config(state='normal')
        else:
            self.train_button.config(state='disabled')

    def _start_model_training(self):
        if self.df is None or not self.numerical_cols and not self.categorical_cols and not self.binary_cols:
            messagebox.showwarning('Cảnh báo', 'Vui lòng tải dữ liệu và tiền xử lý (phân loại biến) ở các tab trước.')
            return
        if 'Heart Attack Risk' not in self.df.columns:
            messagebox.showerror('Lỗi', 'Cột \'Heart Attack Risk\' không tìm thấy. Không thể huấn luyện mô hình.')
            return

        self.status_label.config(text='Đang huấn luyện mô hình, vui lòng chờ...', foreground='blue')
        self._disable_buttons()
        test_size = self.test_size_var.get()
        use_smote = self.use_smote_var.get()
        threading.Thread(target=self._train_model_task, args=(test_size, use_smote)).start()

    def _train_model_task(self, test_size, use_smote):
        try:
            self.model_trainer = ModelTrainer(
                self.df,
                self.numerical_cols,
                self.categorical_cols,
                self.binary_cols,
                target_col='Heart Attack Risk'
            )
            metrics = self.model_trainer.train_model(test_size=test_size, use_smote=use_smote)
            roc_image_data = self.model_trainer.plot_roc_curve()
            cm_image_data = self.model_trainer.plot_confusion_matrix()
            self.app.data_queue.put({
                'status': 'model_trained',
                'metrics': metrics,
                'roc_image': roc_image_data,
                'cm_image': cm_image_data,
                'message': 'Mô hình đã được huấn luyện và đánh giá thành công.'
            })
        except Exception as e:
            self.app.data_queue.put({'status': 'error', 'message': f'Lỗi khi huấn luyện mô hình: {e}'})

    def _update_gui_after_training(self, metrics, roc_image_data, cm_image_data, message):
        self.status_label.config(text=f'Hoàn thành: {message}', foreground='green')
        
        model_name = metrics.get('model_name', 'Unknown')
        metrics_text = (
            f'Mô hình được chọn: {model_name}\n\n'
            f'Accuracy: {metrics["accuracy"]:.4f}\n'
            f'Precision: {metrics["precision"]:.4f}\n'
            f'Recall: {metrics["recall"]:.4f}\n'
            f'F1-Score: {metrics["f1_score"]:.4f}\n'
            f'ROC-AUC: {metrics["roc_auc"]:.4f}\n\n'
            f'Ma trận nhầm lẫn:\n'
            f'  [TN  FP]\n'
            f'  [FN  TP]\n'
            f'  {metrics["confusion_matrix"]}'
        )
        self.metrics_label.config(text=metrics_text)
        if roc_image_data:
            roc_image = Image.open(io.BytesIO(base64.b64decode(roc_image_data)))
            roc_image = roc_image.resize((300, 250), Image.LANCZOS)
            self.roc_photo = ImageTk.PhotoImage(roc_image)
            self.roc_image_label.config(image=self.roc_photo)
            self.roc_image_label.image = self.roc_photo
        if cm_image_data:
            cm_image = Image.open(io.BytesIO(base64.b64decode(cm_image_data)))
            cm_image = cm_image.resize((300, 250), Image.LANCZOS)
            self.cm_photo = ImageTk.PhotoImage(cm_image)
            self.cm_image_label.config(image=self.cm_photo)
            self.cm_image_label.image = self.cm_photo

        self._enable_buttons()
        messagebox.showinfo('Hoàn thành', message)