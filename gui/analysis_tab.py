import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import numpy as np
import os
from datetime import datetime

class AnalysisTab:
    def __init__(self, master, app_instance):
        self.master = master
        self.app = app_instance
        self.model_trainer = None
        self.coef_df = None  
        self.current_fig = None 
        self.frame = ttk.Frame(self.master, padding='15')
        self.frame.pack(expand=True, fill='both')
        ttk.Label(self.frame, text='Phân tích Kết quả Mô hình', font=('Arial', 18, 'bold')).pack(pady=10)
        control_frame = ttk.LabelFrame(self.frame, text='Điều khiển Phân tích', padding='15')
        control_frame.pack(pady=10, padx=10, fill='x')
        self.analyze_button = ttk.Button(control_frame, text='Phân tích Hệ số & Tỷ lệ Odds', command=self._start_analysis)
        self.analyze_button.pack(pady=5)
        export_frame = ttk.Frame(control_frame)
        export_frame.pack(pady=5, fill='x')
        
        self.view_chart_button = ttk.Button(export_frame, text='Xem Biểu đồ (Cửa sổ mới)', command=self._view_chart_new_window)
        self.view_chart_button.pack(side=tk.LEFT, padx=5)
        
        self.export_button = ttk.Button(export_frame, text='Xuất Kết quả', command=self._export_results)
        self.export_button.pack(side=tk.LEFT, padx=5)
        coef_frame = ttk.LabelFrame(self.frame, text='Hệ số & Tỷ lệ Odds của Đặc trưng', padding='15')
        coef_frame.pack(pady=10, padx=10, fill='x', expand=False)
        self.coef_text = tk.Text(coef_frame, height=15, state='disabled', wrap=tk.WORD, font=('Consolas', 9))
        self.coef_text.pack(pady=5, fill='both', expand=True)
        plot_frame = ttk.LabelFrame(self.frame, text='Biểu đồ Tác động của Yếu tố', padding='15')
        plot_frame.pack(pady=10, padx=10, fill='both', expand=True)
        self.canvas_frame = ttk.Frame(plot_frame)
        self.canvas_frame.pack(fill='both', expand=True)
        self.canvas = None
        self.status_label = ttk.Label(self.frame, text='', foreground='blue')
        self.status_label.pack(pady=10)
        self._disable_buttons()

    def update_model_trainer(self, model_trainer_instance):
        self.model_trainer = model_trainer_instance
        if self.model_trainer and self.model_trainer.model:
            self.status_label.config(text='Mô hình đã được huấn luyện. Sẵn sàng phân tích.')
            self._enable_buttons()
        else:
            self.status_label.config(text='Chưa có mô hình để phân tích.', foreground='red')
            self._disable_buttons()
            
    def _disable_buttons(self):
        self.analyze_button.config(state='disabled')
        self.view_chart_button.config(state='disabled')
        self.export_button.config(state='disabled')

    def _enable_buttons(self):
        if self.model_trainer and self.model_trainer.model:
            self.analyze_button.config(state='normal')
        else:
            self.analyze_button.config(state='disabled')
        if self.coef_df is not None:
            self.view_chart_button.config(state='normal')
            self.export_button.config(state='normal')
        else:
            self.view_chart_button.config(state='disabled')
            self.export_button.config(state='disabled')

    def _start_analysis(self):
        if self.model_trainer is None or self.model_trainer.model is None:
            messagebox.showwarning('Cảnh báo', 'Vui lòng huấn luyện mô hình ở Tab \'Mô hình\' trước.')
            return
        if not hasattr(self.model_trainer.model.named_steps.get('classifier'), 'coef_'):
            messagebox.showwarning('Cảnh báo', 'Phân tích hệ số chỉ hỗ trợ cho mô hình Logistic Regression.')
            return

        self.status_label.config(text='Đang phân tích hệ số và tỷ lệ odds...', foreground='blue')
        self._disable_buttons()
        threading.Thread(target=self._analysis_task).start()

    def _analysis_task(self):
        try:
            coef_df = self.model_trainer.get_model_coefficients()
            if coef_df is None or coef_df.empty:
                raise ValueError('Không thể lấy hệ số mô hình. Có thể mô hình chưa được huấn luyện hoặc có lỗi.')

            coef_df['Odds Ratio'] = np.exp(coef_df['Coefficient'])
            
            self.app.data_queue.put({
                'status': 'analysis_done',
                'coef_df': coef_df,
                'message': 'Đã hoàn thành phân tích hệ số và tỷ lệ odds.'
            })
        except Exception as e:
            self.app.data_queue.put({'status': 'error', 'message': f'Lỗi khi phân tích mô hình: {e}'})

    def _update_gui_after_analysis(self, coef_df, message):
        self.coef_df = coef_df
        
        self.status_label.config(text=f'Hoàn thành: {message}', foreground='green')
        self.coef_text.config(state='normal')
        self.coef_text.delete(1.0, tk.END)
        coef_df_sorted = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
        display_text = 'PHÂN TÍCH HỆ SỐ VÀ TỶ LỆ ODDS\n'
        display_text += '=' * 60 + '\n\n'
        
        for _, row in coef_df_sorted.iterrows():
            feature = row['Feature']
            coef = row['Coefficient']
            odds_ratio = row['Odds Ratio']
            if coef > 0:
                impact = 'TĂNG nguy cơ'
                impact_symbol = '↑'
            else:
                impact = 'GIẢM nguy cơ' 
                impact_symbol = '↓'
            
            display_text += f'{impact_symbol} {feature:<30}\n'
            display_text += f'   Hệ số: {coef:>8.4f}\n'
            display_text += f'   Odds Ratio: {odds_ratio:>5.4f}\n'
            if abs(coef) > 0.1:
                display_text += f'   >>> {impact} MẠNH <<<\n'
            elif abs(coef) > 0.05:
                display_text += f'   >> {impact} trung bình <<\n'
            else:
                display_text += f'   > {impact} nhẹ <\n'
            display_text += '-' * 40 + '\n'
        
        display_text += '\nGHI CHÚ:\n'
        display_text += '• Hệ số > 0: Tăng nguy cơ đau tim\n'
        display_text += '• Hệ số < 0: Giảm nguy cơ đau tim\n'
        display_text += '• Odds Ratio > 1: Tăng khả năng\n'
        display_text += '• Odds Ratio < 1: Giảm khả năng\n'
        
        self.coef_text.insert(tk.END, display_text)
        self.coef_text.config(state='disabled')
        self._plot_feature_importance(coef_df)
        self._enable_buttons()
        messagebox.showinfo('Hoàn thành', message)

    def _plot_feature_importance(self, coef_df):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
            plt.close('all')
        sorted_coef = coef_df.sort_values(by='Coefficient', ascending=False)
        top_positive = sorted_coef.head(10)
        top_negative = sorted_coef.tail(10)
        plot_df = pd.concat([top_positive, top_negative]).drop_duplicates().sort_values(by='Coefficient', ascending=True)

        if plot_df.empty:
            messagebox.showwarning('Thông báo', 'Không có đặc trưng nào để vẽ biểu đồ.')
            return
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['red' if c < 0 else 'green' for c in plot_df['Coefficient']]
        bars = ax.barh(plot_df['Feature'], plot_df['Coefficient'], color=colors, alpha=0.7)
        ax.set_xlabel('Hệ số (Coefficient)', fontsize=12)
        ax.set_title('Tác động của các yếu tố đến nguy cơ đau tim\n(Hệ số Logistic Regression)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Tăng nguy cơ'),
                          Patch(facecolor='red', alpha=0.7, label='Giảm nguy cơ')]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        plt.subplots_adjust(left=0.3)
        self.current_fig = fig
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _view_chart_new_window(self):
        '''Hiển thị biểu đồ trong cửa sổ mới'''
        if self.coef_df is None:
            messagebox.showwarning('Cảnh báo', 'Chưa có kết quả phân tích để hiển thị.')
            return
        chart_window = tk.Toplevel(self.master)
        chart_window.title('Biểu đồ Phân tích Kết quả Mô hình')
        chart_window.geometry('1000x700')
        chart_frame = ttk.Frame(chart_window)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        sorted_coef = self.coef_df.sort_values(by='Coefficient', ascending=False)
        top_positive = sorted_coef.head(15)
        top_negative = sorted_coef.tail(15)
        plot_df = pd.concat([top_positive, top_negative]).drop_duplicates().sort_values(by='Coefficient', ascending=True)
        if plot_df.empty:
            messagebox.showwarning('Thông báo', 'Không có đặc trưng nào để vẽ biểu đồ.')
            chart_window.destroy()
            return
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['red' if c < 0 else 'green' for c in plot_df['Coefficient']]
        bars = ax.barh(plot_df['Feature'], plot_df['Coefficient'], color=colors, alpha=0.7)
        ax.set_xlabel('Hệ số (Coefficient)', fontsize=12)
        ax.set_ylabel('Đặc trưng', fontsize=12)
        ax.set_title('Tác động của các yếu tố đến nguy cơ đau tim\n(Hệ số Logistic Regression)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Tăng nguy cơ'),
                          Patch(facecolor='red', alpha=0.7, label='Giảm nguy cơ')]
        ax.legend(handles=legend_elements, loc='lower right')
        for bar, coef in zip(bars, plot_df['Coefficient']):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                   f'{coef:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.3)
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, chart_frame)
        toolbar.update()
        chart_window.lift()
        chart_window.focus_force()

    def _export_results(self):
        '''Xuất kết quả phân tích ra file Excel'''
        if self.coef_df is None:
            messagebox.showwarning('Cảnh báo', 'Chưa có kết quả phân tích để xuất.')
            return
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            file_path = filedialog.asksaveasfilename(
                defaultextension='.xlsx',
                filetypes=[('Excel files', '*.xlsx'), ('CSV files', '*.csv'), ('All files', '*.*')],
                title=f'Lưu kết quả phân tích (Đề xuất: Phan_tich_mo_hinh_{timestamp}.xlsx)',
                initialdir=os.getcwd()
            )
            if not file_path:
                return
            if not file_path.endswith(('.xlsx', '.csv')):
                file_path += '.xlsx'
            
            export_df = self.coef_df.copy()
            export_df = export_df.sort_values(by='Coefficient', key=abs, ascending=False)
            export_df['Tac_dong'] = export_df['Coefficient'].apply(
                lambda x: 'Tăng nguy cơ' if x > 0 else 'Giảm nguy cơ'
            )
            export_df['Muc_do'] = export_df['Coefficient'].apply(
                lambda x: 'Mạnh' if abs(x) > 0.1 else ('Trung bình' if abs(x) > 0.05 else 'Nhẹ')
            )
            summary_data = {
                'Thong_tin': [
                    'Tổng số đặc trưng',
                    'Đặc trưng tăng nguy cơ',
                    'Đặc trưng giảm nguy cơ',
                    'Đặc trưng tác động mạnh nhất',
                    'Hệ số cao nhất',
                    'Hệ số thấp nhất',
                    'Thời gian phân tích'
                ],
                'Gia_tri': [
                    len(export_df),
                    len(export_df[export_df['Coefficient'] > 0]),
                    len(export_df[export_df['Coefficient'] < 0]),
                    export_df.iloc[0]['Feature'],
                    f'{export_df['Coefficient'].max():.4f}',
                    f'{export_df['Coefficient'].min():.4f}',
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            if file_path.endswith('.xlsx'):
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='Chi_tiet_he_so', index=False)
                    summary_df.to_excel(writer, sheet_name='Tong_quan', index=False)
                    guide_data = {
                        'Huong_dan': [
                            'Hệ số > 0: Đặc trưng làm tăng nguy cơ đau tim',
                            'Hệ số < 0: Đặc trưng làm giảm nguy cơ đau tim',
                            'Odds Ratio > 1: Tăng khả năng mắc bệnh',
                            'Odds Ratio < 1: Giảm khả năng mắc bệnh',
                            'Giá trị tuyệt đối càng lớn thì tác động càng mạnh',
                            '',
                            'Phân loại mức độ tác động:',
                            '- Mạnh: |hệ số| > 0.1',
                            '- Trung bình: 0.05 < |hệ số| <= 0.1',
                            '- Nhẹ: |hệ số| <= 0.05'
                        ]
                    }
                    guide_df = pd.DataFrame(guide_data)
                    guide_df.to_excel(writer, sheet_name='Huong_dan', index=False)
                    
            else:
                export_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            messagebox.showinfo('Thành công', f'Đã xuất kết quả ra file:\n{file_path}')
            
        except Exception as e:
            messagebox.showerror('Lỗi', f'Không thể xuất file: {str(e)}')