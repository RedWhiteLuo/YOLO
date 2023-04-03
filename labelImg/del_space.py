import os


def filerename(path_):
    filelist = os.listdir(path_)  # 文件夹中的文件列表
    for file in filelist:  # 逐次遍历文件夹下的文件
        path2 = file.replace(' ', '_')  # 将文件名中的空格替换成下划线,或者替换成其他的也行
        Olddir = os.path.join(path_, file)  # 完整的的文件路径
        Newdir = os.path.join(path_, path2)  # 得到新的路径
        os.rename(Olddir, Newdir)  # 重命名


if __name__ == "__main__":
    filerename("data/Picture")
