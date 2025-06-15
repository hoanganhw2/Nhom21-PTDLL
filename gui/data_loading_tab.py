import tkinter as tk
from tkinter import Misc, ttk, filedialog
import threading
from queue import Queue
import pandas as pd

class DataLoaderTab:
    
    def __init__(self, master: Misc | None = None, data_queue: Queue = None) -> None:
        self.master = master
        self.data_queue = data_queue
        self.frame = ttk.Frame(self.master, padding='10')
        self.frame.pack(expand=True, fill='both')
        control_frame = ttk.LabelFrame(self.frame, text='Tải Dữ liệu CSV', padding='15')
        control_frame.pack(pady=10, padx=10, fill='x')
        self.file_path_label = ttk.Label(control_frame, text='Chưa có file nào được chọn.', wraplength=700)
        self.file_path_label.pack(side=tk.LEFT, padx=(0, 10), fill='x', expand=True)
        self.load_button = ttk.Button(control_frame, text='Chọn file CSV', command=self._load_data_threaded)
        self.load_button.pack(side=tk.RIGHT)
        info_frame = ttk.LabelFrame(self.frame, text='Thông tin Dữ liệu', padding='15')
        info_frame.pack(pady=10, padx=10, fill='both', expand=True)
        self.info_text = tk.Text(info_frame, height=10, state='disabled', wrap=tk.WORD, font=('Consolas', 10))
        self.info_text.pack(pady=5, fill='both', expand=True)
        
        table_frame = ttk.LabelFrame(self.frame, text='Xem trước dữ liệu', padding='15')
        table_frame.pack(pady=10, padx=10, fill='both', expand=True)
        preview_control_frame = ttk.Frame(table_frame)
        preview_control_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(preview_control_frame, text='Số dòng hiển thị:').pack(side='left', padx=(0, 5))
        self.preview_limit_var = tk.StringVar(value='5')
        self.preview_limit_entry = ttk.Entry(preview_control_frame, textvariable=self.preview_limit_var, width=10)
        self.preview_limit_entry.pack(side='left', padx=(0, 10))
        
        self.update_preview_button = ttk.Button(preview_control_frame, text='Cập nhật', command=self._update_preview)
        self.update_preview_button.pack(side='left', padx=(0, 10))
        self.update_preview_button.config(state='disabled')
        
        self.tree = ttk.Treeview(table_frame, show='headings')
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar_y = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        scrollbar_y.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=scrollbar_y.set)
        scrollbar_x = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        scrollbar_x.pack(side='bottom', fill='x')
        self.tree.configure(xscrollcommand=scrollbar_x.set)
    
    def _load_data_threaded(self) -> None:
        file_path = filedialog.askopenfilename(
            title='Chọn file CSV',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if file_path:
            self.file_path_label.config(text=f'Đang tải: {file_path}')
            self.load_button.config(state='disabled')
            threading.Thread(target=self._load_data, args=(file_path,)).start()
            
    def _load_data(self, file_path: str) -> None:
        try:
            df = pd.read_csv(file_path)
            self.data_queue.put({'status': 'success', 'df': df, 'file_path': file_path})
        except Exception as e:
            self.data_queue.put({'status': 'error', 'message': str(e)})
    
    def update_gui_with_data(self, df, file_path) -> None:
        self.df = df
        self.file_path_label.config(text=f'File đã chọn: {file_path}')
        self.load_button.config(state='normal')
        self.update_preview_button.config(state='normal')
        self._display_basic_info(df)
        try:
            limit_text = self.preview_limit_var.get().strip()
            if limit_text == '' or limit_text == '0':
                limit = 5 
            else:
                limit = int(limit_text)
                if limit <= 0:
                    limit = 5
        except ValueError:
            limit = 5
            self.preview_limit_var.set('5')
        
        max_limit = min(limit, len(df))
        self._display_dataframe_preview(df.head(max_limit))
        
    def _display_basic_info(self, df) -> None:
        info = f'Kích thước dữ liệu: {df.shape[0]} hàng, {df.shape[1]} cột\n\n'
        info += 'Số lượng giá trị thiếu mỗi cột:\n'
        missing_values = df.isnull().sum()
        for col, count in missing_values[missing_values > 0].items():
            info += f'- {col}: {count}\n'
        if missing_values.sum() == 0:
            info += 'Không có giá trị thiếu nào.\n'
        info += '\nKiểu dữ liệu của các cột:\n'
        for col, dtype in df.dtypes.items():
            info += f'- {col}: {dtype}\n'
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        self.info_text.config(state='disabled')

    def _display_dataframe_preview(self, df_preview):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree['columns'] = list(df_preview.columns)
        for col in df_preview.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor='center')
        for index, row in df_preview.iterrows():
            self.tree.insert('', 'end', values=list(row))
    
    def _update_preview(self) -> None:
        '''Cập nhật xem trước dữ liệu theo số dòng người dùng nhập'''
        if hasattr(self, 'df') and self.df is not None:
            try:
                limit_text = self.preview_limit_var.get().strip()
                if limit_text == '' or limit_text == '0':
                    limit = 5
                else:
                    limit = int(limit_text)
                    if limit <= 0:
                        limit = 5
                max_limit = min(limit, len(self.df))
                preview_df = self.df.head(max_limit)
                self._display_dataframe_preview(preview_df)
                
            except ValueError:
                preview_df = self.df.head(5)
                self._display_dataframe_preview(preview_df)
                self.preview_limit_var.set('5')