import tkinter as tk
from tkinter import ttk, messagebox
import queue
from gui.data_loading_tab import DataLoaderTab
from gui.preprocessing_tab import PreprocessingTab
from gui.model_training_tab import ModelTrainingTab
from gui.analysis_tab import AnalysisTab
from gui.correlation_tab import CorrelationTab 

class HeartDiseasePredictorApp:
    def __init__(self, master):
        self.master = master
        master.title('Ứng dụng Dự đoán Nguy cơ Đau tim')
        master.geometry('1200x800')
        self.df = None 
        self.numerical_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        self.model_trainer_instance = None 
        self.data_queue = queue.Queue()
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        self._create_tabs()
        self.master.after(100, self._check_queue)

    def _create_tabs(self):
        data_loader_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_loader_frame, text='1. Tải dữ liệu')
        self.data_loader_tab = DataLoaderTab(data_loader_frame, self.data_queue)
        preprocessing_frame = ttk.Frame(self.notebook)
        self.notebook.add(preprocessing_frame, text='2. Tiền xử lý')
        self.preprocessing_tab = PreprocessingTab(preprocessing_frame, self)
        model_training_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_training_frame, text='3. Mô hình')
        self.model_training_tab = ModelTrainingTab(model_training_frame, self)
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text='4. Phân tích')
        self.analysis_tab = AnalysisTab(analysis_frame, self)
        correlation_frame = ttk.Frame(self.notebook)
        self.notebook.add(correlation_frame, text='5. Tương quan')
        self.correlation_tab = CorrelationTab(correlation_frame, self)


    def _check_queue(self):
        try:
            while True:
                task = self.data_queue.get_nowait()
                self._process_task(task)
                self.data_queue.task_done()
        except queue.Empty:
            pass
        self.master.after(100, self._check_queue)

    def _process_task(self, task):
        status = task['status']
        
        match status:
            case 'success':
                self._handle_data_loaded(task)
            case 'preprocessing_done':
                self._handle_preprocessing_done(task)
            case 'model_trained':
                self._handle_model_trained(task)
            case 'analysis_done':
                self._handle_analysis_done(task)
            case 'plot_generated':
                self._handle_plot_generated(task)
            case 'error':
                self._handle_error(task)

    def _handle_data_loaded(self, task):
        self.df = task['df']
        file_path = task['file_path']
        self.data_loader_tab.update_gui_with_data(self.df, file_path)
        self.preprocessing_tab.update_dataframe(self.df)
        messagebox.showinfo('Thông báo', 'Dữ liệu đã được tải thành công!')

    def _handle_preprocessing_done(self, task):
        function_name = task.get('function')
        message = task.get('message', 'Tiền xử lý hoàn tất.')
        df_new = task.get('df')
        results = task.get('results')

        if df_new is not None:
            self.df = df_new

        if function_name == 'classify_variables' and results:
            self.numerical_cols = results['numerical']
            self.categorical_cols = results['categorical']
            self.binary_cols = results['binary']
            self.model_training_tab.update_dataframe_and_cols(
                self.df, self.numerical_cols, self.categorical_cols, self.binary_cols
            )
            self.correlation_tab.update_dataframe_and_cols(
                self.df, self.numerical_cols, self.categorical_cols, self.binary_cols
            )
        
        self.preprocessing_tab._update_gui_after_preprocessing(
            function_name, results=results, df_new=df_new, message=message
        )

    def _handle_model_trained(self, task):
        metrics = task.get('metrics')
        roc_image = task.get('roc_image')
        cm_image = task.get('cm_image')
        message = task.get('message', 'Huấn luyện mô hình hoàn tất.')
        
        self.model_trainer_instance = self.model_training_tab.model_trainer 
        
        self.model_training_tab._update_gui_after_training(
            metrics, roc_image, cm_image, message
        )
        if self.model_trainer_instance:
            self.analysis_tab.update_model_trainer(self.model_trainer_instance)

    def _handle_analysis_done(self, task):
        coef_df = task.get('coef_df')
        message = task.get('message', 'Phân tích mô hình hoàn tất.')
        self.analysis_tab._update_gui_after_analysis(coef_df, message)

    def _handle_plot_generated(self, task):
        plot_type = task.get('plot_type')
        fig_bytes = task.get('fig_bytes')
        message = task.get('message')
        
        canvas_mapping = {
            'corr_matrix': self.correlation_tab.corr_canvas_frame,
            'pair_plot': self.correlation_tab.pair_canvas_frame,
            'target_corr': self.correlation_tab.target_corr_canvas_frame,
            'categorical_analysis': self.correlation_tab.cat_canvas_frame
        }
        
        if plot_type in canvas_mapping:
            self.correlation_tab._display_plot(fig_bytes, canvas_mapping[plot_type], plot_type)
            self.correlation_tab.status_label.config(text=f'Hoàn thành: {message}', foreground='green')

    def _handle_error(self, task):
        messagebox.showerror('Lỗi', f'Có lỗi xảy ra: {task["message"]}')


if __name__ == '__main__':
    root = tk.Tk()
    app = HeartDiseasePredictorApp(root)
    root.mainloop()