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
        master.title("Ứng dụng Dự đoán Nguy cơ Đau tim")
        master.geometry("1200x800")
        self.df = None 
        self.numerical_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        self.model_trainer_instance = None 
        self.data_queue = queue.Queue()
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        self._create_tabs()
        self.master.after(100, self._check_queue)

    def _create_tabs(self):
        # Tab 1: Tải dữ liệu
        data_loader_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_loader_frame, text="1. Tải dữ liệu")
        self.data_loader_tab = DataLoaderTab(data_loader_frame, self.data_queue)

        # Tab 2: Tiền xử lý
        preprocessing_frame = ttk.Frame(self.notebook)
        self.notebook.add(preprocessing_frame, text="2. Tiền xử lý")
        self.preprocessing_tab = PreprocessingTab(preprocessing_frame, self)

        # Tab 3: Mô hình
        model_training_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_training_frame, text="3. Mô hình")
        self.model_training_tab = ModelTrainingTab(model_training_frame, self)

        # Tab 4: Phân tích
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="4. Phân tích")
        self.analysis_tab = AnalysisTab(analysis_frame, self)

        # Tab 5: Tương quan
        correlation_frame = ttk.Frame(self.notebook)
        self.notebook.add(correlation_frame, text="5. Tương quan")
        # Truyền self (instance của HeartDiseasePredictorApp) vào tab tương quan
        self.correlation_tab = CorrelationTab(correlation_frame, self)


    def _check_queue(self):
        """Kiểm tra queue để xem có dữ liệu mới từ thread tải file/tiền xử lý/huấn luyện/phân tích/tương quan không."""
        try:
            while True:
                task = self.data_queue.get_nowait()
                if task["status"] == "success": # Từ DataLoaderTab
                    self.df = task["df"]
                    file_path = task["file_path"]
                    self.data_loader_tab.update_gui_with_data(self.df, file_path)
                    self.preprocessing_tab.update_dataframe(self.df)
                    messagebox.showinfo("Thông báo", "Dữ liệu đã được tải thành công!")
                elif task["status"] == "preprocessing_done": # Từ PreprocessingTab
                    function_name = task.get("function")
                    message = task.get("message", "Tiền xử lý hoàn tất.")
                    df_new = task.get("df")
                    results = task.get("results")

                    if df_new is not None:
                        self.df = df_new # Cập nhật DataFrame toàn cục

                    if function_name == "classify_variables" and results:
                        self.numerical_cols = results["numerical"]
                        self.categorical_cols = results["categorical"]
                        self.binary_cols = results["binary"]
                        # Cập nhật DataFrame và danh sách cột cho tab huấn luyện mô hình và tương quan
                        self.model_training_tab.update_dataframe_and_cols(
                            self.df, self.numerical_cols, self.categorical_cols, self.binary_cols
                        )
                        self.correlation_tab.update_dataframe_and_cols(
                            self.df, self.numerical_cols, self.categorical_cols, self.binary_cols
                        )
                    # Cập nhật GUI của preprocessing tab
                    self.preprocessing_tab._update_gui_after_preprocessing(
                        function_name,
                        results=results,
                        df_new=df_new,
                        message=message
                    )
                elif task["status"] == "model_trained": # Từ ModelTrainingTab
                    metrics = task.get("metrics")
                    roc_image = task.get("roc_image")
                    cm_image = task.get("cm_image")
                    message = task.get("message", "Huấn luyện mô hình hoàn tất.")
                    
                    self.model_trainer_instance = self.model_training_tab.model_trainer # Lưu instance ModelTrainer
                    
                    self.model_training_tab._update_gui_after_training(
                        metrics, roc_image, cm_image, message
                    )
                    # Sau khi mô hình huấn luyện xong, cập nhật cho tab Phân tích
                    if self.model_trainer_instance:
                        self.analysis_tab.update_model_trainer(self.model_trainer_instance)
                elif task["status"] == "analysis_done": # Từ AnalysisTab
                    coef_df = task.get("coef_df")
                    message = task.get("message", "Phân tích mô hình hoàn tất.")
                    self.analysis_tab._update_gui_after_analysis(coef_df, message)
                elif task["status"] == "plot_generated": # Từ CorrelationTab
                    plot_type = task.get("plot_type")
                    fig_bytes = task.get("fig_bytes")
                    message = task.get("message")
                    
                    # Gọi hàm hiển thị plot tương ứng trong CorrelationTab
                    if plot_type == "corr_matrix":
                        self.correlation_tab._display_plot(fig_bytes, self.correlation_tab.corr_canvas_frame, plot_type)
                    elif plot_type == "pair_plot":
                        self.correlation_tab._display_plot(fig_bytes, self.correlation_tab.pair_canvas_frame, plot_type)
                    elif plot_type == "target_corr":
                        self.correlation_tab._display_plot(fig_bytes, self.correlation_tab.target_corr_canvas_frame, plot_type)
                    elif plot_type == "categorical_analysis":
                        self.correlation_tab._display_plot(fig_bytes, self.correlation_tab.cat_canvas_frame, plot_type)
                    
                    self.correlation_tab.status_label.config(text=f"Hoàn thành: {message}", foreground="green")

                elif task["status"] == "error":
                    messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {task['message']}")
                self.data_queue.task_done()
        except queue.Empty:
            pass

        self.master.after(100, self._check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseasePredictorApp(root)
    root.mainloop()