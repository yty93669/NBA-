import os
import cv2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Toplevel
from PIL import Image, ImageTk


def extract_image_features(image_path):
    """提取图像特征 - 处理中文路径问题的主函数"""
    try:
        print(f"尝试读取图像: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"文件不存在: {image_path}")
            return None
        
        # 检查文件大小
        file_size = os.path.getsize(image_path)
        print(f"文件大小: {file_size} bytes")
        if file_size == 0:
            print("文件为空")
            return None
        
        # 方法1：使用numpy读取文件，然后用cv2.imdecode解码
        try:
            # 读取图像文件为字节数组
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # 将字节数据转换为numpy数组
            nparr = np.frombuffer(image_data, np.uint8)
            
            # 使用cv2.imdecode解码图像
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                print(f"cv2.imdecode成功读取图像，尺寸: {image.shape}")
                
                # 使用HOG特征提取
                hog = cv2.HOGDescriptor()
                resized_image = cv2.resize(image, (128, 128))
                features = hog.compute(resized_image)
                
                if features is not None:
                    print(f"成功提取特征，特征维度: {features.shape}")
                    return features.flatten()
            else:
                print("cv2.imdecode解码失败")
                
        except Exception as e:
            print(f"cv2.imdecode方法失败: {e}")
        
        # 方法2：回退到PIL方法
        return extract_image_features_pil_method(image_path)
        
    except Exception as e:
        print(f"提取图像特征时发生错误: {e}")
        return None


def extract_image_features_pil_method(image_path):
    """使用PIL的方法"""
    try:
        pil_image = Image.open(image_path)
        print(f"PIL成功读取图像，模式: {pil_image.mode}, 尺寸: {pil_image.size}")
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        print(f"成功转换为OpenCV格式，图像尺寸: {image.shape}")
        
        hog = cv2.HOGDescriptor()
        resized_image = cv2.resize(image, (128, 128))
        features = hog.compute(resized_image)
        
        if features is not None:
            print(f"成功提取特征，特征维度: {features.shape}")
            return features.flatten()
        else:
            print("HOG特征提取失败")
            return None
            
    except Exception as e:
        print(f"PIL方法失败: {e}")
        return None


def text_search(query, df, top_n=10):
    """文本检索"""
    try:
        vectorizer = TfidfVectorizer()
        text_matrix = vectorizer.fit_transform(df['中文名'] + " " + df['号码'])
        query_vector = vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, text_matrix).flatten()
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        return df.iloc[top_indices]
    except Exception as e:
        print(f"文本检索错误: {e}")
        messagebox.showerror("错误", f"文本检索失败: {e}")
        return pd.DataFrame()


def image_search(image_path, df, top_n=10):
    """图像检索"""
    try:
        print(f"开始图像检索，查询图像: {image_path}")
        query_features = extract_image_features(image_path)
        if query_features is None:
            messagebox.showerror("错误", "无法读取查询图像，请检查图像文件格式和完整性")
            return pd.DataFrame()
        
        print(f"查询图像特征提取成功，开始处理数据库图像")
        all_image_features = []
        valid_indices = []
        
        for idx, img_url in enumerate(df['img_url']):
            img_path = os.path.join(image_folder, os.path.basename(img_url))
            print(f"处理第{idx+1}/{len(df)}张图像: {img_path}")
            features = extract_image_features(img_path)
            if features is not None:
                all_image_features.append(features)
                valid_indices.append(idx)
            else:
                print(f"跳过无效图像: {img_path}")
        
        if len(all_image_features) == 0:
            messagebox.showerror("错误", "数据库中没有有效的图像文件")
            return pd.DataFrame()
        
        print(f"成功处理{len(all_image_features)}张数据库图像")
        
        # 转换为numpy数组并标准化
        all_image_features = np.array(all_image_features)
        all_image_features = normalize(all_image_features, axis=1)
        query_features = normalize(query_features.reshape(1, -1), axis=1)
        
        # 计算相似度
        similarity_scores = cosine_similarity(query_features, all_image_features).flatten()
        
        # 获取最相似的图像索引
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        
        # 映射回原始数据框的索引
        original_indices = [valid_indices[i] for i in top_indices]
        
        print(f"检索完成，返回{len(original_indices)}个结果")
        return df.iloc[original_indices]
        
    except Exception as e:
        print(f"图像检索错误: {e}")
        messagebox.showerror("错误", f"图像检索失败: {e}")
        return pd.DataFrame()


def show_results(results):
    """显示检索结果"""
    result_text.delete(1.0, tk.END)
    if results.empty:
        result_text.insert(tk.END, "没有找到相关结果\n")
        return
        
    for index, row in results.iterrows():
        result_text.insert(tk.END, f"中文名: {row['中文名']}, 号码: {row['号码']}\n")
        result_text.tag_add(str(index), f"{result_text.index('end-2c')} linestart", f"{result_text.index('end-2c')} lineend")
        result_text.tag_bind(str(index), "<Button-1>", lambda event, r=row: show_player_card(r))


def show_player_card(row):
    """显示球员卡"""
    player_card_window = Toplevel(root)
    player_card_window.title(row['中文名'])

    img_path = os.path.join(image_folder, os.path.basename(row['img_url']))
    try:
        image = Image.open(img_path)
        image.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(image)
        image_label = Label(player_card_window, image=photo)
        image_label.image = photo
        image_label.pack(pady=10)
    except Exception as e:
        print(f"加载球员图片失败: {e}")
        error_label = Label(player_card_window, text="无法加载球员图片")
        error_label.pack(pady=10)

    info_text = f"球队: {row['球队']}\n中文名: {row['中文名']}\n英文名: {row['英文名']}\n号码: {row['号码']}\n位置: {row['位置']}\n身高: {row['身高']}\n体重: {row['体重']}\n球龄: {row['球龄']}"
    info_label = Label(player_card_window, text=info_text, justify=tk.LEFT)
    info_label.pack(pady=10)


def search():
    """执行检索"""
    query = entry.get().strip()
    if query:
        print(f"执行文本检索: {query}")
        results = text_search(query, data)
    elif selected_image_path:
        print(f"执行图像检索: {selected_image_path}")
        results = image_search(selected_image_path, data)
    else:
        messagebox.showerror("错误", "请输入文本或选择图像")
        return
    show_results(results)


def select_image():
    """选择图像"""
    global selected_image_path
    file_path = filedialog.askopenfilename(
        title="选择图像文件",
        filetypes=[
            ("所有支持的图像", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("所有文件", "*.*")
        ]
    )
    if file_path:
        print(f"选择的图像文件: {file_path}")
        selected_image_path = file_path
        try:
            image = Image.open(file_path)
            image.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo
            
            # 清空文本输入框
            entry.delete(0, tk.END)
            print("图像预览加载成功")
        except Exception as e:
            print(f"图像预览加载失败: {e}")
            messagebox.showerror("错误", f"无法预览图像: {e}")


def check_paths():
    """检查必要的文件和文件夹是否存在"""
    csv_path = "/nba_players_info_new.csv"
    if not os.path.exists(csv_path):
        messagebox.showerror("错误", f"CSV文件不存在: {csv_path}")
        return False
    
    if not os.path.exists(image_folder):
        messagebox.showerror("错误", f"图像文件夹不存在: {image_folder}")
        return False
    
    return True


# 主程序
if __name__ == "__main__":
    # 图片文件夹路径
    image_folder = r'nba_all_players'
    
    # 检查路径
    if not check_paths():
        exit()
    
    # 加载数据
    try:
        data = pd.read_csv("nba_players_info_new.csv", encoding='GBK')
        print(f"成功加载数据，共{len(data)}条记录")
    except Exception as e:
        messagebox.showerror("错误", f"无法加载CSV文件: {e}")
        exit()
    
    # 创建主窗口
    root = tk.Tk()
    root.title("NBA球员信息检索")
    root.geometry("600x500")
    
    # 添加说明标签
    instruction_label = tk.Label(root, text="请输入球员姓名或号码进行文本检索，或选择图像进行图像检索", 
                                font=("Arial", 10))
    instruction_label.pack(pady=5)
    
    # 输入框
    entry = tk.Entry(root, width=50, font=("Arial", 12))
    entry.pack(pady=10)
    
    # 选择图像按钮
    select_image_button = tk.Button(root, text="选择图像", command=select_image, 
                                   font=("Arial", 10))
    select_image_button.pack(pady=5)
    
    # 显示选择的图像
    image_label = tk.Label(root, text="未选择图像", bg="lightgray", width=15, height=8)
    image_label.pack(pady=5)
    
    # 检索按钮
    search_button = tk.Button(root, text="检索", command=search, 
                             font=("Arial", 12), bg="lightblue")
    search_button.pack(pady=10)
    
    # 结果显示文本框
    result_text = tk.Text(root, height=15, width=70, font=("Arial", 10))
    result_text.pack(pady=10)
    
    selected_image_path = None
    
    print("程序启动成功")
    root.mainloop()