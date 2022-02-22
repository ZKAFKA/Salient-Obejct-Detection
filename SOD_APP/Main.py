import tkinter as tk
from tkinter import filedialog
import os
import time
import numpy as np
import skimage
import torch
from PIL import Image
from skimage import io
from skimage.transform import resize
from model.csnet import build_model


#######################################################################################################################
class SOD:

    def __init__(self):
        # 窗体------------------------------------------------------------------------------------
        self.window = tk.Tk()
        self.window.title("图像前景主体分割工具")
        sw = self.window.winfo_screenwidth()
        sh = self.window.winfo_screenheight()
        pos_x = (sw - 400) / 2
        pos_y = (sh - 300) / 2
        self.window.geometry("400x300+%d+%d" % (pos_x, pos_y))

        # 文件目录------------------------------------------------------------------------------------
        self.open_fpath = tk.StringVar()
        self.open_label = tk.Label(self.window, text='打开目录', width=9, height=1)
        self.open_button = tk.Button(self.window, text='选择', width=5, height=1, command=self.openfile)
        self.openPath_entry = tk.Entry(textvariable=self.open_fpath, width=35)

        self.save_fpath = tk.StringVar()
        self.save_label = tk.Label(self.window, text='保存目录', width=9, height=1)
        self.save_button = tk.Button(self.window, text='选择', width=5, height=1, command=self.savefile)
        self.savePath_entry = tk.Entry(textvariable=self.save_fpath, width=35)

        self.open_label.place(x=9, y=13)
        self.open_button.place(x=340, y=10)
        self.openPath_entry.place(x=80, y=13)
        self.save_label.place(x=9, y=53)
        self.save_button.place(x=340, y=50)
        self.savePath_entry.place(x=80, y=53)

        # 日志信息------------------------------------------------------------------------------------
        self.text_frame = tk.Frame(self.window, width=50, height=10)
        self.scroll = tk.Scrollbar(self.text_frame)
        self.log_Text = tk.Text(self.text_frame, width=50, height=10)
        self.text_frame.place(x=21, y=90)
        self.log_Text.pack(side=tk.LEFT, fill=tk.Y)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll.config(command=self.log_Text.yview)
        self.log_Text.config(yscrollcommand=self.scroll.set)

        # 运行按钮
        self.run_button = tk.Button(self.window, text='RUN', width=10, height=2, bg="PINK", command=self.run)
        self.run_button.place(x=160, y=236)

        # 启动
        self.window.mainloop()

    # 文件打开与存放--------------------------------------------------------------------------------
    def openfile(self):
        file_path = tk.filedialog.askdirectory()
        self.open_fpath.set(file_path)

    def savefile(self):
        file_path = tk.filedialog.askdirectory()
        self.save_fpath.set(file_path)

    # 程序运行-------------------------------------------------------------------------------------

    def run(self):
        self.log_Text.delete(1.0, tk.END)
        self.predict(self.open_fpath.get(), self.save_fpath.get())

    def print_log(self, log_message):
        self.log_Text.insert(tk.END, log_message)
        self.log_Text.update()

    #######################################################################################################################
    # 测试
    def predict(self, open_fpath, save_fpath):
        model = build_model(basic_split=[0.5, 0.5], expand=2.0)
        model.cuda()

        this_checkpoint = "Model.pth.tar"

        if os.path.isfile(this_checkpoint):
            self.print_log("=> loading model '{}' \n\n".format(this_checkpoint))
            checkpoint = torch.load(this_checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            self.test(model, open_fpath, save_fpath)
        else:
            self.print_log(this_checkpoint + "Not found.\n")

    def test(self, model, open_fpath, save_fpath):
        model.eval()
        self.print_log("Start predicting.\n\n")
        sal_save_dir = save_fpath
        os.makedirs(sal_save_dir, exist_ok=True)
        img_dir = os.path.join(open_fpath)
        img_list = os.listdir(img_dir)
        count = 0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        with torch.no_grad():
            for img_name in img_list:
                img = skimage.img_as_float(
                    io.imread(os.path.join(img_dir, img_name)))
                # 将非三通道图片转为三通道图片
                if img.shape[2] == 4:
                    img = Image.open(os.path.join(img_dir, img_name), 'r').convert("RGB")
                    img = np.array(img)
                h, w = img.shape[:2]
                img = resize(img, (224, 224),
                             mode='reflect',
                             anti_aliasing=False)
                tmp = (img - mean) / std
                img = np.transpose(tmp, (2, 0, 1))
                img = torch.unsqueeze(torch.FloatTensor(img), 0)
                input_var = torch.autograd.Variable(img)
                input_var = input_var.cuda()

                idx = img_list.index(img_name)
                self.print_log("正在处理第{}张图片... \n".format(idx + 1))
                T1 = time.time()

                predict = model(input_var)

                T2 = time.time()
                self.print_log("处理完成，耗时{}秒\n\n".format((T2 - T1)))

                predict = predict[0]
                predict = torch.sigmoid(predict.squeeze(0).squeeze(0))
                predict = predict.data.cpu().numpy()
                predict = (resize(
                    predict, (h, w), mode='reflect', anti_aliasing=False) *
                           255).astype(np.uint8)
                save_file = os.path.join(sal_save_dir, img_name[0:-4] + '.png')
                io.imsave(save_file, predict)
                count += 1

                # 合成背景图
                img1 = io.imread(os.path.join(img_dir, img_name))
                img2 = predict
                # img2 = io.imread(save_file)
                img_nobg = img1.copy()
                for i in range(h):
                    for j in range(w):
                        if img2[i, j] < 90:
                            img_nobg[i, j, 0] = 255
                            img_nobg[i, j, 1] = 255
                            img_nobg[i, j, 2] = 255
                save_img_nobg = os.path.join(sal_save_dir, img_name[0:-4] + '_withoutBG.png')
                io.imsave(save_img_nobg, img_nobg)

        self.print_log('Dataset: {}, {} images\n'.format(open_fpath, len(img_list)))


if __name__ == "__main__":
    SOD()
