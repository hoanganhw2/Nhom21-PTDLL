import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import pandas as pd
import numpy as np

from core.data_processor import DataProcessor

class PreprocessingTab:
    def __init__(self, master, app_instance) -> None:
        self.master = master
        self.app = app_instance
        self.df = None
        self.frame = ttk.Frame(self.master, padding='15')
        self.frame.pack(expand=True, fill='both')
        self.numerical_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        bp_frame = ttk.LabelFrame(self.frame, text='1. Xử lý Cột Huyết áp', padding='15')
        bp_frame.pack(pady=10, padx=10, fill='x')
        ttk.Label(bp_frame, text='Tách cột \'Blood Pressure\' thành \'Systolic\' và \'Diastolic\'.').pack(anchor='w', pady=5)
        self.bp_button = ttk.Button(bp_frame, text='Thực hiện tách Huyết áp', command=self._start_bp_processing)
        self.bp_button.pack(pady=5)
        classify_frame = ttk.LabelFrame(self.frame, text='2. Phân loại Biến', padding='15')
        classify_frame.pack(pady=10, padx=10, fill='x')
        ttk.Label(classify_frame, text='Phân loại các biến thành số, phân loại, nhị phân.').pack(anchor='w', pady=5)
        self.classify_button = ttk.Button(classify_frame, text='Phân loại Biến', command=self._start_classify_variables)
        self.classify_button.pack(pady=5)
        self.var_info_text = tk.Text(classify_frame, height=8, state='disabled', wrap=tk.WORD, font=('Consolas', 9))
        self.var_info_text.pack(pady=5, fill='x', expand=True)
        missing_frame = ttk.LabelFrame(self.frame, text='3. Xử lý Giá trị Thiếu', padding='15')
        missing_frame.pack(pady=10, padx=10, fill='x')
        ttk.Label(missing_frame, text='Chọn chiến lược điền giá trị thiếu:').pack(side=tk.LEFT, padx=(0, 10))
        self.missing_strategy_var = tk.StringVar(value='mean')
        ttk.Radiobutton(missing_frame, text='Trung bình (Numerical)', variable=self.missing_strategy_var, value='mean').pack(side=tk.LEFT)
        ttk.Radiobutton(missing_frame, text='Trung vị (Numerical)', variable=self.missing_strategy_var, value='median').pack(side=tk.LEFT)
        ttk.Radiobutton(missing_frame, text='Phổ biến nhất (All)', variable=self.missing_strategy_var, value='most_frequent').pack(side=tk.LEFT)
        self.handle_missing_button = ttk.Button(missing_frame, text='Xử lý Giá trị Thiếu', command=self._start_handle_missing)
        self.handle_missing_button.pack(pady=5, anchor='e')
        outlier_frame = ttk.LabelFrame(self.frame, text='4. Xử lý Ngoại lai (Isolation Forest)', padding='15')
        outlier_frame.pack(pady=10, padx=10, fill='x')
        ttk.Label(outlier_frame, text='Mức độ nhiễm (Contamination):').pack(side=tk.LEFT, padx=(0, 5))
        self.contamination_var = tk.DoubleVar(value=0.05)
        ttk.Scale(outlier_frame, from_=0.01, to=0.1,
                  variable=self.contamination_var, orient='horizontal').pack(side=tk.LEFT, expand=True, fill='x')
        ttk.Label(outlier_frame, textvariable=self.contamination_var).pack(side=tk.LEFT, padx=(5, 0))
        self.handle_outlier_button = ttk.Button(outlier_frame, text='Xử lý Ngoại lai', command=self._start_handle_outliers)
        self.handle_outlier_button.pack(pady=5, anchor='e')
        self.status_label = ttk.Label(self.frame, text='', foreground='blue')
        self.status_label.pack(pady=10)


    def update_dataframe(self, df) -> None:
        self.df = df.copy()
        self.status_label.config(text='Dữ liệu đã được tải. Sẵn sàng tiền xử lý.')
        self._enable_buttons()

    def _disable_buttons(self) -> None:
        self.bp_button.config(state='disabled')
        self.classify_button.config(state='disabled')
        self.handle_missing_button.config(state='disabled')
        self.handle_outlier_button.config(state='disabled')

    def _enable_buttons(self) -> None:
        self.bp_button.config(state='normal')
        self.classify_button.config(state='normal')
        self.handle_missing_button.config(state='normal')
        self.handle_outlier_button.config(state='normal')

    def _display_var_info(self) -> None:
        info = 'Biến số (Numerical):\n' + ', '.join(self.numerical_cols) + '\n\n'
        info += 'Biến phân loại (Categorical):\n' + ', '.join(self.categorical_cols) + '\n\n'
        info += 'Biến nhị phân (Binary):\n' + ', '.join(self.binary_cols)
        self.var_info_text.config(state='normal')
        self.var_info_text.delete(1.0, tk.END)
        self.var_info_text.insert(tk.END, info)
        self.var_info_text.config(state='disabled')

    def _start_bp_processing(self) -> None:
        if self.df is None:
            messagebox.showwarning('Cảnh báo', 'Vui lòng tải dữ liệu ở Tab \'Tải dữ liệu\' trước.')
            return
        self.status_label.config(text='Đang xử lý cột huyết áp...')
        self._disable_buttons()
        threading.Thread(target=self._process_bp_task).start()

    def _process_bp_task(self) -> None:
        try:
            processor = DataProcessor(self.df)
            processed_df = processor.process_blood_pressure()
            self.app.data_queue.put({'status': 'preprocessing_done',
                                     'function': 'bp_processing',
                                     'df': processed_df,
                                     'message': 'Đã tách thành công Systolic và Diastolic.'})
        except Exception as e:
            self.app.data_queue.put({'status': 'error', 'message': f'Lỗi xử lý huyết áp: {e}'})

    def _start_classify_variables(self) -> None:
        if self.df is None:
            messagebox.showwarning('Cảnh báo', 'Vui lòng tải dữ liệu ở Tab \'Tải dữ liệu\' trước.')
            return
        self.status_label.config(text='Đang phân loại biến...')
        self._disable_buttons()
        threading.Thread(target=self._classify_variables_task).start()

    def _classify_variables_task(self) -> None:
        try:
            processor = DataProcessor(self.df)
            classification_results = processor.classify_variables()
            self.app.data_queue.put({'status': 'preprocessing_done',
                                     'function': 'classify_variables',
                                     'results': classification_results,
                                     'message': 'Đã phân loại biến thành công.'})
        except Exception as e:
            self.app.data_queue.put({'status': 'error', 'message': f'Lỗi phân loại biến: {e}'})

    def _start_handle_missing(self) -> None:
        if self.df is None:
            messagebox.showwarning('Cảnh báo', 'Vui lòng tải dữ liệu ở Tab \'Tải dữ liệu\' trước.')
            return
        if not (self.numerical_cols or self.categorical_cols):
             messagebox.showwarning('Cảnh báo', 'Vui lòng phân loại biến trước khi xử lý giá trị thiếu.')
             return

        strategy = self.missing_strategy_var.get()
        self.status_label.config(text=f'Đang xử lý giá trị thiếu bằng chiến lược \'{strategy}\'...')
        self._disable_buttons()
        threading.Thread(target=self._handle_missing_task, args=(strategy,)).start()

    def _handle_missing_task(self, strategy) -> None:
        try:
            processor = DataProcessor(self.df) 
            processed_df = processor.handle_missing_values(strategy=strategy,
                                                           numerical_cols=self.numerical_cols,
                                                           categorical_cols=self.categorical_cols)
            self.app.data_queue.put({'status': 'preprocessing_done',
                                     'function': 'handle_missing',
                                     'df': processed_df,
                                     'message': f'Đã xử lý giá trị thiếu bằng \'{strategy}\'.'})
        except Exception as e:
            self.app.data_queue.put({'status': 'error', 'message': f'Lỗi xử lý giá trị thiếu: {e}'})

    def _start_handle_outliers(self) -> None:
        if self.df is None:
            messagebox.showwarning('Cảnh báo', 'Vui lòng tải dữ liệu ở Tab \'Tải dữ liệu\' trước.')
            return
        if not self.numerical_cols:
             messagebox.showwarning('Cảnh báo', 'Không có biến số nào để xử lý ngoại lai. Vui lòng phân loại biến trước.')
             return

        contamination = self.contamination_var.get()
        self.status_label.config(text=f'Đang xử lý ngoại lai với contamination={contamination}...')
        self._disable_buttons()
        threading.Thread(target=self._handle_outliers_task, args=(contamination,)).start()

    def _handle_outliers_task(self, contamination) -> None:
        try:
            processor = DataProcessor(self.df)
            processed_df = processor.handle_outliers_isolation_forest(
                numerical_cols=self.numerical_cols,
                contamination=contamination
            )
            self.app.data_queue.put({'status': 'preprocessing_done',
                                     'function': 'handle_outliers',
                                     'df': processed_df,
                                     'message': f'Đã xử lý ngoại lai với contamination={contamination}.'})        
        except Exception as e:
            self.app.data_queue.put({'status': 'error', 'message': f'Lỗi xử lý ngoại lai: {e}'})

    def _update_gui_after_preprocessing(self, function_name, results=None, df_new=None, message='') -> None:
        self.status_label.config(text=f'Hoàn thành: {message}', foreground='green')
        if df_new is not None:
            self.df = df_new 
            self.app.df = df_new 
            self.app.data_loader_tab.update_gui_with_data(self.df, 'Đã tiền xử lý')
        
        if function_name == 'classify_variables' and results:
            self.numerical_cols = results['numerical']
            self.categorical_cols = results['categorical']
            self.binary_cols = results['binary']
            self._display_var_info()
            messagebox.showinfo('Hoàn thành', f'Đã phân loại biến thành công. Vui lòng kiểm tra lại nếu cần.')
        else:
            messagebox.showinfo('Hoàn thành', message)
        
        # Truyền dữ liệu và phân loại biến sang model training tab nếu đã có phân loại
        if self.df is not None and self.numerical_cols and (self.categorical_cols or self.binary_cols):
            self.app.model_training_tab.update_dataframe_and_cols(
                self.df, self.numerical_cols, self.categorical_cols, self.binary_cols
            )
        
        self._enable_buttons()