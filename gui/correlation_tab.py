import io
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np

class CorrelationTab:
    def __init__(self, master, app_instance):
        self.master = master
        self.app = app_instance
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        self.target_col = 'Heart Attack Risk'

        self.frame = ttk.Frame(self.master, padding='15')
        self.frame.pack(expand=True, fill='both')
        ttk.Label(self.frame, text='Phân tích Tương quan Dữ liệu', font=('Arial', 18, 'bold')).pack(pady=10)
        self.plot_notebook = ttk.Notebook(self.frame)
        self.plot_notebook.pack(expand=True, fill='both', padx=10, pady=10)
        self.corr_matrix_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.corr_matrix_frame, text='Ma trận Tương quan')
        self._setup_corr_matrix_tab()
        self.pair_plot_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.pair_plot_frame, text='Biểu đồ cặp')
        self._setup_pair_plot_tab()
        self.target_corr_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.target_corr_frame, text='Tương quan với mục tiêu')
        self._setup_target_corr_tab()
        self.categorical_analysis_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.categorical_analysis_frame, text='Phân tích biến phân loại')
        self._setup_categorical_analysis_tab()
        self.status_label = ttk.Label(self.frame, text='', foreground='blue')
        self.status_label.pack(pady=10)
        self.plot_notebook.tab(0, state='disabled')
        self.plot_notebook.tab(1, state='disabled')
        self.plot_notebook.tab(2, state='disabled')
        self.plot_notebook.tab(3, state='disabled')


    def update_dataframe_and_cols(self, df, numerical_cols, categorical_cols, binary_cols):
        self.df = df
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols
        self.status_label.config(text='Dữ liệu và phân loại biến đã sẵn sàng cho phân tích tương quan.')
        self._enable_tabs()
        self._plot_correlation_matrix()
        self._plot_target_correlation()
        self._populate_categorical_dropdown()

    def _enable_tabs(self):
        self.plot_notebook.tab(0, state='normal')
        self.plot_notebook.tab(1, state='normal')
        self.plot_notebook.tab(2, state='normal')
        self.plot_notebook.tab(3, state='normal')

    def _setup_corr_matrix_tab(self):
        control_frame = ttk.Frame(self.corr_matrix_frame, padding=5)
        control_frame.pack(side=tk.TOP, fill='x')
        ttk.Button(control_frame, text='Vẽ Ma trận Tương quan', command=self._start_plot_corr_matrix).pack(pady=5)
        self.corr_canvas_frame = ttk.Frame(self.corr_matrix_frame)
        self.corr_canvas_frame.pack(fill='both', expand=True)
        self.corr_canvas = None

    def _setup_pair_plot_tab(self):
        control_frame = ttk.Frame(self.pair_plot_frame, padding=5)
        control_frame.pack(side=tk.TOP, fill='x')
        ttk.Label(control_frame, text='Chọn số lượng cột để vẽ (có thể chậm với nhiều cột):').pack(side=tk.LEFT, padx=5)
        self.pair_plot_num_cols_var = tk.IntVar(value=5)
        ttk.Spinbox(control_frame, from_=2, to=self.app.df.shape[1] if self.app.df is not None else 10,
                    textvariable=self.pair_plot_num_cols_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text='Vẽ Biểu đồ cặp', command=self._start_plot_pair_plot).pack(side=tk.LEFT, padx=5)
        self.pair_canvas_frame = ttk.Frame(self.pair_plot_frame)
        self.pair_canvas_frame.pack(fill='both', expand=True)
        self.pair_canvas = None

    def _setup_target_corr_tab(self):
        control_frame = ttk.Frame(self.target_corr_frame, padding=5)
        control_frame.pack(side=tk.TOP, fill='x')
        ttk.Button(control_frame, text='Vẽ Tương quan với biến mục tiêu', command=self._start_plot_target_corr).pack(pady=5)
        self.target_corr_canvas_frame = ttk.Frame(self.target_corr_frame)
        self.target_corr_canvas_frame.pack(fill='both', expand=True)
        self.target_corr_canvas = None

    def _setup_categorical_analysis_tab(self):
        control_frame = ttk.Frame(self.categorical_analysis_frame, padding=5)
        control_frame.pack(side=tk.TOP, fill='x')

        ttk.Label(control_frame, text='Chọn biến phân loại:').pack(side=tk.LEFT, padx=5)
        self.selected_cat_var = tk.StringVar()
        self.cat_dropdown = ttk.Combobox(control_frame, textvariable=self.selected_cat_var, state='readonly')
        self.cat_dropdown.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text='Vẽ Biểu đồ phân loại', command=self._start_plot_categorical_analysis).pack(side=tk.LEFT, padx=5)
        self.cat_canvas_frame = ttk.Frame(self.categorical_analysis_frame)
        self.cat_canvas_frame.pack(fill='both', expand=True)
        self.cat_canvas = None

    def _populate_categorical_dropdown(self):
        all_categorical_cols = sorted(list(set(self.categorical_cols + self.binary_cols)))
        if self.target_col in all_categorical_cols:
            all_categorical_cols.remove(self.target_col) 
        self.cat_dropdown['values'] = all_categorical_cols
        if all_categorical_cols:
            self.selected_cat_var.set(all_categorical_cols[0])
        else:
            self.selected_cat_var.set('')

    def _start_plot_corr_matrix(self):
        if self.df is None or not self.numerical_cols:
            messagebox.showwarning('Cảnh báo', 'Vui lòng tải dữ liệu và phân loại biến trước.')
            return
        self.status_label.config(text='Đang vẽ ma trận tương quan...', foreground='blue')
        threading.Thread(target=self._plot_correlation_matrix_task).start()

    def _plot_correlation_matrix_task(self):
        try:
            fig_bytes = self._plot_correlation_matrix()
            self.app.data_queue.put({'status': 'plot_generated',
                                     'plot_type': 'corr_matrix',
                                     'fig_bytes': fig_bytes,
                                     'message': 'Ma trận tương quan đã được vẽ.'})
        except Exception as e:
            self.app.data_queue.put({'status': 'error', 'message': f'Lỗi khi vẽ ma trận tương quan: {e}'})

    def _plot_correlation_matrix(self):
        '''Vẽ ma trận tương quan cho các biến số.'''
        if not self.numerical_cols:
            return None
        numeric_df = self.df[self.numerical_cols].select_dtypes(include=np.number)
        if numeric_df.empty:
            messagebox.showwarning('Cảnh báo', 'Không tìm thấy cột số nào để vẽ ma trận tương quan.')
            return None
        self._clear_canvas(self.corr_canvas, self.corr_canvas_frame)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax, linewidths=.5)
        ax.set_title('Ma trận Tương quan của các biến số')
        plt.tight_layout()

        fig_bytes = self._figure_to_bytes(fig)
        self._display_plot(fig_bytes, self.corr_canvas_frame, 'corr_matrix')
        return fig_bytes


    def _start_plot_pair_plot(self):
        if self.df is None or not self.numerical_cols:
            messagebox.showwarning('Cảnh báo', 'Vui lòng tải dữ liệu và phân loại biến trước.')
            return
        num_cols = self.pair_plot_num_cols_var.get()
        if num_cols < 2:
            messagebox.showwarning('Cảnh báo', 'Vui lòng chọn ít nhất 2 cột để vẽ biểu đồ cặp.')
            return
        cols_to_plot = self.numerical_cols
        if len(self.numerical_cols) > num_cols:
            cols_to_plot = np.random.choice(self.numerical_cols, num_cols, replace=False).tolist()

        self.status_label.config(text=f'Đang vẽ biểu đồ cặp cho {len(cols_to_plot)} cột...', foreground='blue')
        threading.Thread(target=self._plot_pair_plot_task, args=(cols_to_plot,)).start()

    def _plot_pair_plot_task(self, cols_to_plot):
        try:
            if not cols_to_plot:
                raise ValueError('Không có cột số nào để vẽ biểu đồ cặp.')

            self._clear_canvas(self.pair_canvas, self.pair_canvas_frame) 
            if len(cols_to_plot) > 5 and len(self.df) > 1000: 
                sample_df = self.df[cols_to_plot].sample(n=1000, random_state=42)
            else:
                sample_df = self.df[cols_to_plot].copy()

            if self.target_col in self.df.columns:
                if self.target_col not in sample_df.columns:
                    sample_df[self.target_col] = self.df[self.target_col]
                g = sns.pairplot(sample_df, hue=self.target_col, diag_kind='kde')
            else:
                g = sns.pairplot(sample_df, diag_kind='kde')
            
            g.figure.suptitle(f'Biểu đồ cặp của {len(cols_to_plot)} biến số (sample: {len(sample_df)} hàng)', y=1.02) 

            fig_bytes = self._figure_to_bytes(g.figure) 
            self.app.data_queue.put({'status': 'plot_generated',
                                     'plot_type': 'pair_plot',
                                     'fig_bytes': fig_bytes,
                                     'message': 'Biểu đồ cặp đã được vẽ.'})
            plt.close(g.figure)

        except Exception as e:
            self.app.data_queue.put({'status': 'error', 'message': f'Lỗi khi vẽ biểu đồ cặp: {e}'})


    def _start_plot_target_corr(self):
        if self.df is None or not self.numerical_cols or self.target_col not in self.df.columns:
            messagebox.showwarning('Cảnh báo', 'Vui lòng tải dữ liệu, phân loại biến và đảm bảo có cột mục tiêu.')
            return
        self.status_label.config(text='Đang vẽ tương quan với biến mục tiêu...', foreground='blue')
        threading.Thread(target=self._plot_target_correlation_task).start()

    def _plot_target_correlation_task(self):
        try:
            fig_bytes = self._plot_target_correlation()
            self.app.data_queue.put({'status': 'plot_generated',
                                     'plot_type': 'target_corr',
                                     'fig_bytes': fig_bytes,
                                     'message': 'Tương quan với biến mục tiêu đã được vẽ.'})
        except Exception as e:
            self.app.data_queue.put({'status': 'error', 'message': f'Lỗi khi vẽ tương quan với mục tiêu: {e}'})

    def _plot_target_correlation(self):
        '''Vẽ biểu đồ thanh tương quan của các biến số với biến mục tiêu.'''
        if not self.numerical_cols or self.target_col not in self.df.columns:
            return None
        numeric_df = self.df[self.numerical_cols + [self.target_col]].select_dtypes(include=np.number)
        if self.target_col not in numeric_df.columns: 
            return None
        correlations = numeric_df.corr()[self.target_col].drop(self.target_col).sort_values(ascending=False)
        self._clear_canvas(self.target_corr_canvas, self.target_corr_canvas_frame)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=correlations.values, y=correlations.index, ax=ax)
        ax.set_title(f'Tương quan của biến số với \'{self.target_col}\'')
        ax.set_xlabel('Hệ số Tương quan')
        ax.set_ylabel('Đặc trưng')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        fig_bytes = self._figure_to_bytes(fig)
        self._display_plot(fig_bytes, self.target_corr_canvas_frame, 'target_corr')
        return fig_bytes


    def _start_plot_categorical_analysis(self):
        selected_cat_col = self.selected_cat_var.get()
        if self.df is None or not selected_cat_col:
            messagebox.showwarning('Cảnh báo', 'Vui lòng chọn một biến phân loại.')
            return
        if self.target_col not in self.df.columns:
            messagebox.showwarning('Cảnh báo', 'Không tìm thấy cột mục tiêu \'Heart Attack Risk\'.')
            return

        self.status_label.config(text=f'Đang vẽ biểu đồ phân tích biến phân loại \'{selected_cat_col}\'...', foreground='blue')
        threading.Thread(target=self._plot_categorical_analysis_task, args=(selected_cat_col,)).start()

    def _plot_categorical_analysis_task(self, cat_col):
        try:
            self._clear_canvas(self.cat_canvas, self.cat_canvas_frame)
            if self.df[self.target_col].dtype == 'int64' or self.df[self.target_col].dtype == 'float64':
                avg_risk_per_cat = self.df.groupby(cat_col)[self.target_col].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=avg_risk_per_cat.index, y=avg_risk_per_cat.values, ax=ax)
                ax.set_title(f'Tỷ lệ Nguy cơ Đau tim theo \'{cat_col}\'')
                ax.set_xlabel(cat_col)
                ax.set_ylabel('Tỷ lệ Nguy cơ Đau tim (Trung bình)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                fig_bytes = self._figure_to_bytes(fig)

                self.app.data_queue.put({'status': 'plot_generated',
                                        'plot_type': 'categorical_analysis',
                                        'fig_bytes': fig_bytes,
                                        'message': f'Biểu đồ phân tích \'{cat_col}\' đã được vẽ.'})
            else: 
                raise ValueError('Cột mục tiêu không phải dạng số. Không thể tính tỷ lệ trung bình.')

        except Exception as e:
            self.app.data_queue.put({'status': 'error', 'message': f'Lỗi khi vẽ biểu đồ phân loại: {e}'}) 
            
    def _clear_canvas(self, canvas, canvas_frame):
        for widget in canvas_frame.winfo_children():
            widget.destroy()
        canvas = None
        plt.close('all')

    def _figure_to_bytes(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return buf.getvalue()

    def _display_plot(self, fig_bytes, parent_frame, plot_type):
        for widget in parent_frame.winfo_children():
            widget.destroy()
        _ = plt.figure() 
        img = Image.open(io.BytesIO(fig_bytes))
        frame_width = max(parent_frame.winfo_width() - 20, 400)
        frame_height = max(parent_frame.winfo_height() - 20, 300)
        
        img = img.resize((frame_width, frame_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        if plot_type == 'corr_matrix':
            self.corr_photo = photo
        elif plot_type == 'pair_plot':
            self.pair_photo = photo
        elif plot_type == 'target_corr':
            self.target_corr_photo = photo
        elif plot_type == 'categorical_analysis':
            self.cat_photo = photo

        label = ttk.Label(parent_frame, image=photo)
        label.image = photo 
        label.pack(expand=True, fill='both')
        if plot_type == 'corr_matrix':
            self.corr_canvas = label
        elif plot_type == 'pair_plot':
            self.pair_canvas = label
        elif plot_type == 'target_corr':
            self.target_corr_canvas = label
        elif plot_type == 'categorical_analysis':
            self.cat_canvas = label

        self.status_label.config(text=f'Hoàn thành: {plot_type.replace('_', ' ').title()} đã được vẽ.', foreground='green')