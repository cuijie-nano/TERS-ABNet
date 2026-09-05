import os
import pickle
import argparse
import numpy as np
import torch
from models.attention_unet import AttUnet
from scipy.stats import multivariate_normal
from skimage import feature, measure
from c.bindings import match_template_pool
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def plot_ters_map(ters_map, save_path='ters_map.tif', dpi=300, normalize = True):
    """
    绘制并保存160张TERS图（逆时针旋转90度并在图片内部添加白色文字，无背景）
    """
    # 创建画布：16行10列
    fig, axes = plt.subplots(10, 16, figsize=(8.0,5.0))   # 16/2=8
    
    # 展平 axes 数组，方便循环遍历
    axes_flat = axes.flatten()
    
    # 生成文字标签
    labels = []
    
    # 前100张图：0-2000
    for i in range(100):
        start = i * 20
        end = start + 20
        labels.append(f'{start}-{end}')
    
    # 后60张图：2800-4000
    for i in range(60):
        start = 2800 + i * 20
        end = start + 20
        labels.append(f'{start}-{end}')
    
    for i in range(160):
        # 取出第 i 张图并转为 numpy 数组
        #i=min(ii,159)
        if hasattr(ters_map[i], 'detach'):
            img = ters_map[i].detach().cpu().numpy()
        else:
            img = ters_map[i]
        
        # 逆时针旋转90度
        img_rotated = np.rot90(img, k=1)
        
        # 绘图
        if normalize ==True:
            im = axes_flat[i].imshow(
                img_rotated,
                cmap='viridis',
                vmin=0,
                vmax=1
            )
        else:
            im = axes_flat[i].imshow(img_rotated, cmap='viridis')
        axes_flat[i].axis('off')
        
        # 获取图片尺寸
        height, width = img_rotated.shape
        
        # 在图片内部上方中间添加文字
        text_x = width / 2
        text_y = height * 0.15  # 位置可以调整，0.9更靠上，0.8更靠下
        
        # 添加白色文字，无背景
        axes_flat[i].text(
            text_x, text_y, labels[i],
            fontsize=4.5,  # 稍微调大字号
            color='white',
            ha='center',
            va='center',
            fontweight='bold'
            # 注意：这里没有bbox参数，所以没有背景
        )
    
    # 调整布局
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    
    # 保存为高分辨率TIFF文件
    plt.savefig(
        save_path, 
        format='tiff', 
        dpi=dpi, 
        bbox_inches='tight', 
        pad_inches=0.05,
        facecolor='white'
    )
    print(f"TERS mapping saved at: {save_path}")
    
    plt.show(block=False)
    plt.pause(0.1)
def test_example(
    model,
    data,
    ):
    
    prediction = model(data)

    return prediction

def find_gaussian_peaks(pos_dist, box_borders, match_threshold=0.7, std=0.3, method='mad'):
    '''
    Find real-space positions of gaussian peaks in a 3D position distribution grid.

    Arguments:
        pos_dist: np.ndarray of torch.Tensor of shape (n_batch, n_x, n_y, n_z). Position distribution.
        box_borders: tuple ((x_start, y_start, z_start), (x_end, y_end, z_end)). Real-space extent of the
            distribution grid in angstroms.
        match_threshold: float. Detection threshold for matching. Regions above the threshold are chosen for
            method 'zncc', and regions below the threshold are chosen for methods 'mad', 'msd, 'mad_norm', and
            'msd_norm'.
        std: float. Standard deviation of peaks to search for in angstroms.
        method: 'zncc', 'mad', 'msd', 'mad_norm', or 'msd_norm. Matching method to use. Either zero-normalized
            cross correlation ('zncc'), mean absolute distance ('mad'), mean squared distance ('msd'), or the
            normalized version of the latter two ('mad_norm', 'msd_norm').

    Returns: xyzs, match, labels
        xyzs: list of np.ndarray or torch.Tensor of shape (num_atoms, 3). Positions of the found atoms.
            Each item in theclist corresponds one batch item.
        matches: np.ndarray or torch.Tensor of same shape as input pos_dist. Array of matching values.
            For method 'zncc' larger values, and for 'mad', 'msd', 'mad_norm', and 'msd_norm' smaller
            values correspond to better match.
        labels: np.ndarray or torch.Tensor of same shape as input pos_dist. Labelled regions where
            match is better than match_threshold.
    '''
    '''
    if method not in ['zncc', 'mad', 'msd', 'mad_norm', 'msd_norm']:
        raise ValueError(f'Unknown matching method `{method}`.')
    '''

    xyzs, matches, labels = _find_peaks_cpu(pos_dist.detach().numpy(), box_borders,
        match_threshold, std, method)
    xyzs = [torch.from_numpy(xyz).type(pos_dist.dtype) for xyz in xyzs]
    matches = torch.from_numpy(matches).type(pos_dist.dtype)
    labels = torch.from_numpy(labels).type(pos_dist.dtype)

    return xyzs, matches, labels
def _find_peaks_cpu(pos_dist, box_borders, match_threshold, std, method):

    n_xyz = pos_dist.shape[1:]
    res = [(box_borders[1][i] - box_borders[0][i]) / n_xyz[i] for i in range(3)]
    pos_dist[pos_dist < 1e-4] = 0 # Very small values cause instabilities in ZNCC values
    xyz_start = [box_borders[0][i] + res[i]/2 for i in range(3)]

    # Create reference gaussian peak to compare against
    r = 3 * std + 1e-6
    r = [r - (r % res[i]) for i in range(3)]
    x_ref, y_ref, z_ref = [np.arange(-r[i], r[i]+1e-6, res[i]) for i in range(3)]
    ref_grid = np.stack(np.meshgrid(x_ref, y_ref, z_ref, indexing='ij'), axis=-1)
    ref_peak = multivariate_normal.pdf(ref_grid, mean=[0, 0, 0], cov=std**2)

    # Match the reference gaussian peak shape with the position distributions
    if method in ['mad', 'msd', 'mad_norm', 'msd_norm']:
        matches = match_template_pool(pos_dist, ref_peak, method=method)
    else:
        matches = []
        for d in pos_dist:
            matches.append(
                feature.match_template(d, ref_peak, pad_input=True, mode='constant', constant_values=0)
            )
        matches = np.stack(matches, axis=0)

    # Threshold the match map
    if method == 'zncc':
        threshold_masks = matches > match_threshold
    else:
        threshold_masks = matches < match_threshold

    # Loop over batch items to label matches and find atom positions
    xyzs = []
    labels = []
    for match, threshold_mask in zip(matches, threshold_masks):
        
        # Label connected regions
        labels_, num_atoms = measure.label(threshold_mask, return_num=True)

        # Loop over labelled regions to find atom positions
        xyzs_ = []
        for target_label in range(1, num_atoms+1):

            # Find best matching xyz position from the labelled region
            match_masked = np.ma.array(match, mask=(labels_ != target_label))
            best_ind = match_masked.argmax() if method == 'zncc' else match_masked.argmin()
            best_ind = np.unravel_index(best_ind, match_masked.shape)
            xyz = [xyz_start[i] + res[i] * best_ind[i] for i in range(3)]

            xyzs_.append(xyz)
            #print(target_label,xyzs_)
        xyzs.append(np.array(xyzs_))
        labels.append(labels_)
    
    labels = np.stack(labels, axis=0)

    return xyzs, matches, labels

def list_tensor2array_atom(xyzs):    
    result = []
    # 对于每个 tensor 添加对应的第四列
    for i, xyz in enumerate(xyzs):
        # 将 tensor 转换为 numpy 数组
        xyz_array = xyz.numpy()
        if xyz_array.shape[0] != 0:
            # 创建第四列
            if i == 0:
                fourth_column = np.full((xyz_array.shape[0], 1), 6)  # 第一个 tensor，第四列是 6
            elif i == 1:
                fourth_column = np.full((xyz_array.shape[0], 1), 7)  # 第二个 tensor，第四列是 7
            elif i == 2:
                fourth_column = np.full((xyz_array.shape[0], 1), 8)  # 第三个 tensor，第四列是 8
            elif i == 3:
                fourth_column = np.full((xyz_array.shape[0], 1), 1)  # 第四个 tensor，第四列是 1
            # 将前三列和第四列拼接在一起
            result.append(np.hstack((xyz_array, fourth_column)))
    # 将所有的结果连接成一个大的二维 array
    if result !=[]:
       # print(result)
       final_result = np.vstack(result)
    else:
       final_result = 0
    return final_result 

def list_tensor2array_group(xyzs):    
    result = []
    # 对于每个 tensor 添加对应的第四列
    for i, xyz in enumerate(xyzs):
        # 将 tensor 转换为 numpy 数组
        xyz_array = xyz.numpy()
        if xyz_array.shape[0] != 0:
            # 创建第四列

            fourth_column = np.full((xyz_array.shape[0], 1), i)  # 第一个 tensor，第四列是 6

            # 将前三列和第四列拼接在一起
            result.append(np.hstack((xyz_array, fourth_column)))
    # 将所有的结果连接成一个大的二维 array
    if result !=[]:
       # print(result)
       final_result = np.vstack(result)
    else:
       final_result = 0
    return final_result    

def array2image_marged_atom(data, savefile):
    # 创建一个二维数组，元素介于0-1之间
    # data
    # 将数组的值映射到0-255范围内
    data[data<0]=0

    # 创建红色通道的图像
    channel = np.zeros((128, 128, 3))
    mmax = np.max(data[3,:,:])
    if mmax == 0:
        mmax = 1
    channel[:, :, 0] = data[2,:,:] + (data[3,:,:])**2/mmax
    channel[:, :, 1] = data[0,:,:] + (data[3,:,:])**2/mmax
    channel[:, :, 2] = data[1,:,:] + (data[3,:,:])**2/mmax
    '''
    if np.max(channel[:,:,0]) >0.1:
        channel[:,:,0] = (channel[:,:,0]/np.max(channel[:,:,0])*255)
    if np.max(channel[:,:,1]) >0.1:
        channel[:,:,1] = (channel[:,:,1]/np.max(channel[:,:,1])*255)
    if np.max(channel[:,:,2]) >0.1:
        channel[:,:,2] = (channel[:,:,2]/np.max(channel[:,:,2])*255)
    '''

    channel = channel * 255
    channel[channel>255] = 255
    channel = channel.astype(np.uint8)

    # 将红色通道的图像转换为PIL图像对象
    red_image = Image.fromarray(channel)
    # 将图像保存为TIFF文件
    red_image = red_image.rotate(90, expand=True)

    red_image.save(savefile)
    print(f"Atom map saved at: {savefile}")

    # 显示图像，但不阻塞程序
    plt.figure(figsize=(6, 6))
    plt.imshow(red_image)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.1)
    
def array2image_marged_bond(data, savefile):
    # 创建一个二维数组，元素介于0-1之间
    # data
    # 将数组的值映射到0-255范围内
    data[data<0]=0

    # 创建红色通道的图像
    channel = np.zeros((128, 128, 3))
    color = color_set()
    for i in range(9): 
        channel[:, :, 0] = channel[:, :, 0] + data[i,:,:] * color[i][0]
        channel[:, :, 1] = channel[:, :, 1] + data[i,:,:] * color[i][1]
        channel[:, :, 2] = channel[:, :, 2] + data[i,:,:] * color[i][2]
    '''
    if np.max(channel[:,:,0]) >0.1:
        channel[:,:,0] = (channel[:,:,0]/np.max(channel[:,:,0])*255)
    if np.max(channel[:,:,1]) >0.1:
        channel[:,:,1] = (channel[:,:,1]/np.max(channel[:,:,1])*255)
    if np.max(channel[:,:,2]) >0.1:
        channel[:,:,2] = (channel[:,:,2]/np.max(channel[:,:,2])*255)
    '''

    channel = channel * 255
    channel[channel>255] = 255
    channel = channel.astype(np.uint8)

    # 将红色通道的图像转换为PIL图像对象
    red_image = Image.fromarray(channel)
    # 将图像保存为TIFF文件
    red_image = red_image.rotate(90, expand=True)
    red_image.save(savefile)
    print(f"Bond map saved at: {savefile}")

    # 显示图像，但不阻塞程序
    plt.figure(figsize=(6, 6))
    plt.imshow(red_image)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.1)


    
def color_set():
    # 根据您的要求分配19种颜色
    colors_rgb = [
        # 第1-2个：绿色系
        (255, 255, 255),     # 1. 森林绿
       
        
        # 第3-5个：蓝色系
        (90, 90, 255),     # 3. 钴蓝
        (192, 192, 255),    # 4. 皇家蓝
        
        
        # 第6个：红色系
        (255, 180, 180),     # 6. 亮红
        
        # 第7-9个：绿色系
        (0,255,0),       # 7. 纯绿
        
        
        # 第10-12个：蓝色系
        (0,225,225),    # 10. 道奇蓝
       
        
        # 第13-14个：红色系
        (225,225/3,0),     # 13. 砖红
      
        
        # 第18个：蓝色系
        (0,0,255),       # 18. 海军蓝
        
        # 第19个：红蓝色系（紫色）
        (225,0,225)      # 19. 紫色（红蓝混合）
    ]
    
    # 转换为0-1之间的值
    colors_normalized = [(r/255.0, g/255.0, b/255.0) for r, g, b in colors_rgb]
    
    return colors_normalized

def color_set_g():
    # 根据您的要求分配19种颜色
    colors_rgb = [
        # 第1-2个：绿色系
        (225/1.5, 255/1.5, 225/1.5),     # 1. 森林绿
     
        
        # 第3-5个：蓝色系
        (192/2, 192/2, 255/1.7),     # 3. 钴蓝
        (192/2, 192/2, 255/1.7),    # 4. 皇家蓝
       
        
        # 第6个：红色系
        (255/1.7, 180/1.7, 180/1.7),     # 6. 亮红
        
        # 第7-9个：绿色系
        (0,255/2,0),       # 7. 纯绿
      
        
        # 第10-12个：蓝色系
        (0, 150/2, 255/2),     # 16. 深蓝绿
       
        # 第13-14个：红色系
        (225,225/3,0),     # 13. 砖红

        
        # 第18个：蓝色系
        (0,0,255/1.5),       # 18. 海军蓝
          # 第19个：红蓝色系（紫色）
        (225/1.5,0,225/1.5)      # 19. 紫色（红蓝混合）
   
    ]
    
    # 转换为0-1之间的值
    colors_normalized = [(r/255.0, g/255.0, b/255.0) for r, g, b in colors_rgb]
    
    return colors_normalized

def plot_pos(coord_prid, file_name, size = 20):   # 20pm
    # 示例的 coord_prid 矩阵
    # 假设 n = 10, 每行表示 [x, y, z, element]
    lenth = 12.5*2
    # 提取 x, y 坐标和元素类型
    x = coord_prid[:, 0]+lenth/2
    y = coord_prid[:, 1]+lenth/2
    elements = coord_prid[:, 3]
    
    # 设置颜色映射
    color_map = {6: 'gray', 7: 'blue', 8: 'red', 1: 'lightgray'}
    colors = [color_map[element] for element in elements]
    
    # 创建绘图
    fig, ax = plt.subplots()
    
    # 绘制圆点
    for i in range(len(x)):
        circle = plt.Circle((x[i], y[i]), size, color=colors[i], fill=True)
        ax.add_artist(circle)
    
    # 设置坐标轴范围和比例
    
    ax.set_xlim(0,lenth)
    ax.set_ylim(0,lenth)
    ax.set_aspect('equal', 'box')
    
    # 隐藏坐标轴
    #ax.axis('off')
    ax.set_xticks([]);
    ax.set_yticks([])
    # 显示图形
    plt.savefig(file_name,format='svg')  
    print(f"Atom position saved at: {file_name}")

    plt.show(block=False)
    plt.pause(0.1)
    
def connect5(coord_prid, pos_group_prid=None, out=3,use_group = False):
    # coord_prid  n*4, 原子乘以4    x y r element
    # graph n*n, 原子乘以原子
    N = len(coord_prid)
    graph = np.zeros((1,N,N))
    #scale = 8.33333*2/128*100    #pixel -> pm
    #coord_prid = coord_prid0 * 100 #A -> pm
    #coord_prid[:,2:] = coord_prid[:,2:]/100
    #----------------------------------------------step1  根据基团连接
    absent = [[]]
    if use_group == True:
        if pos_group_prid.all() == None:
            print('group not used, pos_group_prid is needed')
        else:
            '''
            group_list = ['[CH3]',
                          '[CH2]',
                          '[CH]',
                            '[cH]',
                            '[NH2]',
                            '[NH]',
                            '[nH]',
                            '[OH]',
                            'C-C',
                            'C=C',
                            'C#C',
                            'C-N',
                            'C=N',
                            'C#N',
                            'C-O',
                            'C=O',
                            'n1cccc1', #C4N
                            'o1cccc1', #C4O
                            'c1ccccc1',#C6  
                            'n1ccccc1',#C5N 
                            ]
            '''
            group_zone = [1.09*1.5,
                          1.01*1.5,
                          1.01*1.5,
                          0.98*1.5,
                          1.54*1.5,
                          1.48*1.5,
                          1.43*1.5,
                          1.42*1.6,
                          1.42*1.6]
                          
            # atom_num = [4,3,2,2,3,2,2,2,2,2,2,2,2,2,2,2,5,5,6,6]
            atom_num = [#{6:1,1:3},
                        #{6:1,1:2},
                        {6:1,1:1},
                        {7:1,1:2},
                        {7:1,1:1},
                        {8:1,1:1},
                        {6:2},
                        {6:1,7:1},
                        {6:1,8:1},
                        {7:2},
                        {7:1,8:1}]
            atom_in_group,absent = find_nearest_neighbors(pos_group_prid, coord_prid, knn=atom_num, rcut=group_zone)
            #print(atom_in_group)
            #print(absent)
            bond_type = [[1],[1],[1],[1],[1,2,3],[1,2,3],[1,2],[1,2,3],[1,2]]
            n_graph_last=1
            n_graph=1
            for i, group in enumerate(pos_group_prid):
                if  absent[i]!=[]:         #如果是环并且存在缺失，则跳过
                    continue
                if len(atom_in_group[i]) == 2:
                    n_graph_last = n_graph
                    n_graph = np.shape(graph)[0]
                    #print(n_graph_last,n_graph,'\n')
                    n_repeat = len(bond_type[int(group[3])])
                    graph = np.tile(graph, (n_repeat, 1, 1))
                    for kk,order in enumerate(bond_type[int(group[3])]):
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][0],atom_in_group[i][1]] = order
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][1],atom_in_group[i][0]] = order
                        
                if len(atom_in_group[i]) == 3:
                    n_graph_last = n_graph
                    n_graph = np.shape(graph)[0]
                    n_repeat = len(bond_type[int(group[3])])
                    graph = np.tile(graph, (n_repeat, 1, 1))
                    for kk,order in enumerate(bond_type[int(group[3])]):
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][0],atom_in_group[i][1]] = order
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][1],atom_in_group[i][0]] = order                  
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][0],atom_in_group[i][2]] = order
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][2],atom_in_group[i][0]] = order   
                    
                if len(atom_in_group[i]) == 4:
                    n_graph_last = n_graph
                    n_graph = np.shape(graph)[0]
                    n_repeat = len(bond_type[int(group[3])])
                    graph = np.tile(graph, (n_repeat, 1, 1))
                    for kk,order in enumerate(bond_type[int(group[3])]):
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][0],atom_in_group[i][1]] = order
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][1],atom_in_group[i][0]] = order                  
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][0],atom_in_group[i][2]] = order
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][2],atom_in_group[i][0]] = order 
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][0],atom_in_group[i][3]] = order
                        graph[kk*n_graph:kk*n_graph+n_graph,atom_in_group[i][3],atom_in_group[i][0]] = order 
                        

    #draw_molecule_2d(coord_prid, graph)
    #----------------------------------------------step2  根据原子补充连接
    # 示例邻接矩阵和原子类型
    '''
    graph = np.array([
        [0, 1, 0, 0],
        [1, 0, 1.5, 0],
        [0, 1.5, 0, 1],
        [0, 0, 1, 0]
    ], dtype=float)
    '''

    atom_types = coord_prid[:,-1].astype(int).tolist()
    valence_dict = {6: 4, 7:3, 8: 2, 1:1}        #原子：饱和配位数

    idx = [] #配位数过高的graph索引
    score = [] #打分，有几个原子配位不饱和就是几分
    for item, graphi in enumerate(graph):
        unsaturate_atoms = saturate_atoms2(graphi, valence_dict, atom_types) #大于0代表缺，小于0代表配位高了
        score.append(sum(np.abs(unsaturate_atoms))) #有几个原子配位不饱和就是几分

           # draw_molecule_2d(coord_prid, graphi)
    #graph_list1 = [graph_list[i] for i in range(0, len(graph_list), 1) if i not in idx] # 删除后的列表
    # draw_molecule_2d(coord_prid, graph)
    return graph,absent,score  

def find_nearest_neighbors(pos_group_prid, coord_prid, knn, rcut):
    # 发现基团位置为中心，基团大概包含的原子，这里不去看是否在半径内。
    nearest_neighbors = []
    absent = []                #n个元素，每个元素存储在第n个基团范围内，缺少的原子数，以便在下次迭代时找到。
    is_whole = True
    for i in range(len(pos_group_prid)):
        absent_i = []
        pos = pos_group_prid[i, :2]  # 取出pos_group_prid的第i行的前两列作为中心点坐标
        distances = cdist([pos], coord_prid[:, :2])[0]  # 计算pos与coord_prid中所有点的距离
        #print(distances)
        #print(distances)
        # valid = np.sum(distances < rcut[int(pos_group_prid[i,3])] )
        near = np.argsort(distances)                           # 按照距离排序
        atom_num0 = knn[int(pos_group_prid[i,3])] 
        #print(atom_num0)
        #print(atom_num0)
        nearest_indices = np.empty((0,))
        #print(len(atom_num0))
        for j in range(len(atom_num0)):
            #print(j)
            #以CH3为例， 取离基团i最近的C元素1个，以及最近的H元素3个
            #print(near)
            nearest_indices0 = near[np.in1d(coord_prid[near,3], list(atom_num0.keys())[j])][:list(atom_num0.values())[j]] #基团的原子标号
            #if i==5:
                #print(nearest_indices0)
                #print(len(atom_num0))
            nearest_indices0 = nearest_indices0[distances[nearest_indices0] < rcut[int(pos_group_prid[i,3])]]  #只保留满足rcut的原子
            #if i==5:
                #print(nearest_indices0)
            num_matchedAtom = np.sum(distances[nearest_indices0] < rcut[int(pos_group_prid[i,3])])
            #print(rcut[int(pos_group_prid[i,3])])
            #print(num_matchedAtom)
            #print(list(atom_num0.values())[j])
            if  num_matchedAtom < list(atom_num0.values())[j]:
                num_absent = list(atom_num0.values())[j]-num_matchedAtom
                absent_i.append([ list(atom_num0.keys())[j] ] * num_absent)
            #nearest_indices0 = near[coord_prid[near,3] in list(atom_eum0.keys())[j]][:list(atom_eum0.values())[j]]   
            nearest_indices = np.concatenate((nearest_indices, nearest_indices0),axis = 0) 
                                     # 筛选出near中最近的某种元素
        #nearest_indices = np.argsort(distances)[:knn[int(pos_group_prid[i,3])] ]  # 找到距离最近的两个点的索引
        #nearest_indices = nearest_indices[:valid]
            nearest_indices = nearest_indices.astype(int)
        nearest_neighbors.append(nearest_indices)
        absent.append(absent_i)
    return nearest_neighbors, absent

def saturate_atoms2(graph, valence_dict, atom_types):
    """
    判断原子配位饱和状态，并补充单键、双键或三键使所有原子饱和。
    
    :param graph: 邻接矩阵 (numpy array)
    :param valence_dict: 原子的最大配位数字典
    :param atom_types: 每个原子的类型列表
    :return: 更新后的邻接矩阵
    """
    num_atoms = len(graph)
    
    # 计算每个原子的当前配位数
    current_valences = np.sum(graph, axis=1)
    
    # 获取每个原子的最大配位数
    max_valences = np.array([valence_dict[atom] for atom in atom_types])
    
    # 判断未饱和的原子
    unsaturated_atoms = [(max_valences[i] - current_valences[i]) 
                         for i in range(num_atoms) ]
    
    return unsaturated_atoms

def plot_grouppos_line(coord_prid,pos_group_prid,file_name, size = 20)  :   # 20pm
    # 示例的 coord_prid 矩阵
    # 假设 n = 10, 每行表示 [x, y, z, element]

    group_zone = [1.09*1.5,
                  1.01*1.5,
                  1.01*1.5,
                  0.98*1.5,
                  1.54*1.5,
                  1.48*1.5,
                  1.43*1.6,
                  1.42*1.6,
                  1.42*1.6]
                  
    # atom_num = [4,3,2,2,3,2,2,2,2,2,2,2,2,2,2,2,5,5,6,6]
    atom_num = [#{6:1,1:3},
                #{6:1,1:2},
                {6:1,1:1},
                {7:1,1:2},
                {7:1,1:1},
                {8:1,1:1},
                {6:2},
                {6:1,7:1},
                {6:1,8:1},
                {7:2},
                {7:1,8:1}]
    atom_in_group,absent = find_nearest_neighbors(pos_group_prid, coord_prid, knn=atom_num, rcut=group_zone)
    # 提取 x, y 坐标和元素类型
    lenth = 12.5*2
    coord_prid = coord_prid[:, 0:2]+lenth/2
    # y = coord_prid[:, 1]+lenth/2
    
    
    # 设置颜色映射
    color = color_set_g()

    #colors = {1:'gray', 5:'red',  17:'blue', 4:'lightgray'}
    
    # 创建绘图
    fig, ax = plt.subplots()
    
    for i, atom_pair in enumerate(atom_in_group):
        group_idx = pos_group_prid[i,3]
        plt.plot([coord_prid[int(atom_pair[0]),0],coord_prid[int(atom_pair[1]),0] ], 
                 [coord_prid[int(atom_pair[0]),1],coord_prid[int(atom_pair[1]),1] ],
                  color=color[int(group_idx)], linewidth=1)
    
    
    # 设置坐标轴范围和比例
    
    ax.set_xlim(0,lenth)
    ax.set_ylim(0,lenth)
    ax.set_aspect('equal', 'box')
    
    # 隐藏坐标轴
    #ax.axis('off')
    ax.set_xticks([]);
    ax.set_yticks([])
    # 显示图形
    plt.savefig(file_name,format='svg')  
    print(f"Bond position saved at: {file_name}")

    plt.show(block=False)
    plt.pause(0.1)    
def gen_sdfv3000_withoutsmiles(coord_prid, graph, score,smiles_true, n, filename):
        # 示例分子
    #mol1 = Chem.MolFromSmiles(smiles_true)  # 乙醇
    #fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=1024)
    

    """
    生成 V3000 格式的 SDF 文件
    coord_prid: (N,4) array, xyz + atomic number
    graph: list of m (N,N) arrays, 每个是键连接矩阵
    n: int, 取前 n 个结构
    filename: 输出 SDF 文件名
    """
    for struct_idx in range(min(n, len(graph))):
        graph_i = graph[struct_idx]   #zeors NN
        graph_i[graph_i == 1.5] = 4
    N = coord_prid.shape[0]
    atomic_num_to_symbol = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl',
        35: 'Br', 53: 'I'
        # 需要的话可扩展
    }

    def atom_block():
        lines = []
        lines.append("M  V30 BEGIN ATOM")
        for idx, row in enumerate(coord_prid, start=1):
            x, y, z, Z = row
            symbol = atomic_num_to_symbol.get(int(Z), f"[{int(Z)}]")
            lines.append(f"M  V30 {idx} {symbol} {x:.4f} {y:.4f} {z:.4f} 0")
        lines.append("M  V30 END ATOM")
        return lines

    def bond_block(adj_matrix):
        lines = []
        lines.append("M  V30 BEGIN BOND")
        bond_id = 1
        for i in range(N):
            for j in range(i+1, N):
                order = int(adj_matrix[i, j])
                if order > 0:
                    lines.append(f"M  V30 {bond_id} {order} {i+1} {j+1}")
                    bond_id += 1
        lines.append("M  V30 END BOND")
        return lines

    sdf_lines = []
    
    idxx = min(range(len(score)), key=lambda i: abs(score[i]))
    adj_matrix = np.array(graph[idxx])
    # 分子标题行
    score_i = score[idxx]
    smiles_i = 'smiles'#convert_to_smiles(coord_prid[:,-1].tolist(), graph[idxx].tolist()) 
    
    #mol2 = Chem.MolFromSmiles(smiles_i)  # 乙醛
    #fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=1024)
    similarity = 00#DataStructs.BulkTanimotoSimilarity(fp1, [fp2])[0]
    similaritydice =00#DataStructs.DiceSimilarity(fp1, fp2)
    
    sdf_lines.append(f"Structure_{idxx+1}")
    sdf_lines.append(f"Score \t {score_i}, smilesprid \t"+smiles_i + "\t smilestrue \t"+smiles_true + f"\t similartanimoto \t{similarity}" + f"\t similardice \t{similaritydice}")  # 程序/用户信息行
    sdf_lines.append("")  # 空行
    sdf_lines.append(f"  {N:>3}  {np.sum(np.triu(adj_matrix)>0):>3}  0  0  0  0            999 V3000")
    sdf_lines.append("M  V30 BEGIN CTAB")
    num_bonds = int(np.sum(np.triu(adj_matrix) > 0))
    sdf_lines.append(f"M  V30 COUNTS {N} {num_bonds} 0 0 0")
    sdf_lines.extend(atom_block())
    sdf_lines.extend(bond_block(adj_matrix))
    sdf_lines.append("M  V30 END CTAB")
    sdf_lines.append("M  END")
    sdf_lines.append(f">  <ID>")
    sdf_lines.append(str(idxx+1))
    sdf_lines.append("$$$$")
        
    for struct_idx in range(min(n, len(graph))):
        adj_matrix = np.array(graph[struct_idx])
        # 分子标题行
        score_i = score[struct_idx]
        #smiles_i = convert_to_smiles(coord_prid[:,-1].tolist(), graph[struct_idx].tolist()) 
        
        #mol2 = Chem.MolFromSmiles(smiles_i)  # 乙醛
        #fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=1024)
        similarity = 00#DataStructs.BulkTanimotoSimilarity(fp1, [fp2])[0]
        similaritydice =00#DataStructs.DiceSimilarity(fp1, fp2)
        sdf_lines.append(f"Structure_{struct_idx+1}")
        sdf_lines.append(f"Score \t {score_i}, smilesprid \t"+smiles_i + "\t smilestrue \t"+smiles_true + f"\t similartanimoto \t{similarity}" + f"\t similardice \t{similaritydice}")  # 程序/用户信息行
        sdf_lines.append("")  # 空行
        sdf_lines.append(f"  {N:>3}  {np.sum(np.triu(adj_matrix)>0):>3}  0  0  0  0            999 V3000")
        sdf_lines.append("M  V30 BEGIN CTAB")
        num_bonds = int(np.sum(np.triu(adj_matrix) > 0))
        sdf_lines.append(f"M  V30 COUNTS {N} {num_bonds} 0 0 0")
        sdf_lines.extend(atom_block())
        sdf_lines.extend(bond_block(adj_matrix))
        sdf_lines.append("M  V30 END CTAB")
        sdf_lines.append("M  END")
        sdf_lines.append(f">  <ID>")
        sdf_lines.append(str(struct_idx+1))
        sdf_lines.append("$$$$")

    with open(filename, "w") as f:
        f.write("\n".join(sdf_lines))
#%%
# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Train TERS-ABNet"
    )

    # Data
    parser.add_argument(
        "--savedir",
        type=str,
        default="../examples/1_plane_mole/",
    )
        
    parser.add_argument(
        "--input_file",
        type=str,
        default="../examples/1_plane_mole/ters_map.pt",
    )

    # Model_A
    parser.add_argument(
        "--out_channels_a",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--x_channels_a",
        nargs="+",
        type=int,
        default=[64, 64, 96],
    )
    parser.add_argument(
        "--q_channels_a",
        nargs="+",
        type=int,
        default=[64, 64, 64],
    )
    parser.add_argument(
        "--attention_type_a",
        type=str,
        default="softmax",
        choices=["softmax", "sigmoid"],
    )
    parser.add_argument(
        "--model_dir_a",
        type=str,
        default="../checkpoints/ANet_base/model_100.pth",
    )
    
    # Model_B
    parser.add_argument(
        "--out_channels_b",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--x_channels_b",
        nargs="+",
        type=int,
        default=[64, 64, 96],
    )
    parser.add_argument(
        "--q_channels_b",
        nargs="+",
        type=int,
        default=[64, 64, 64],
    )
    parser.add_argument(
        "--attention_type_b",
        type=str,
        default="sigmoid",
        choices=["softmax", "sigmoid"],
    )
    parser.add_argument(
        "--model_dir_b",
        type=str,
        default="../checkpoints/BNet_base/model_100.pth",
    )
    parser.add_argument(
        "--geom",
        action="store_true",
    )
    
    args = parser.parse_args()

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    os.makedirs(
        args.savedir,
        exist_ok=True,
    )

    # Save arguments
    with open(
        os.path.join(args.savedir, "args.pkl"),
        "wb",
    ) as f:
        pickle.dump(args, f)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model_A = AttUnet(
        out_channels=args.out_channels_a,
        x_channels=args.x_channels_a,
        q_channels=args.q_channels_a,
        attention_type=args.attention_type_a,
        )
    model_A.load_state_dict(torch.load(args.model_dir_a))
    model_A.eval()

    model_B = AttUnet(
        out_channels=args.out_channels_b,
        x_channels=args.x_channels_b,
        q_channels=args.q_channels_b,
        attention_type=args.attention_type_b,
        )
    model_B.load_state_dict(torch.load(args.model_dir_b))
    model_B.eval()
    
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    data = torch.load(args.input_file)
    plot_ters_map(data[0,:,:,:], save_path=args.savedir+'ters_map_same_scale.tif',normalize = True)
    plot_ters_map(data[0,:,:,:], save_path=args.savedir+'ters_map.tif',normalize = False)
    # --------------------------------------------------------
    # Prediction 
    # --------------------------------------------------------
    atom_map = test_example(
                            model_A,
                            data,
                            )
      
    bond_map = test_example(
                            model_B,
                            data,
                            ) 

    array2image_marged_atom(atom_map[0,:,:,:].detach().numpy(), args.savedir+'atom_map.tif')
    array2image_marged_bond(bond_map[0,:,:,:].detach().numpy(), args.savedir+'bond_map.tif')

    if args.geom == True:
        box_borders = ((-12.5-12.5/128, -12.5-12.5/128, -0.1), (12.5-12.5/128, 12.5-12.5/128, 0.1))
    
        pos_dist1 = atom_map[0,:,:,:].unsqueeze(-1)
        xyzs1, matches, labels=find_gaussian_peaks(pos_dist1, box_borders, match_threshold=0.80, std=0.20, method='msd_norm')
        atom_coord_prid= list_tensor2array_atom(xyzs1)[:, [0,1,2,3]]
        np.savetxt(args.savedir+"atom_coord.txt", atom_coord_prid, fmt="%.6f",delimiter="\t")      # 制表符分隔
    
        pos_dist2 = bond_map[0,:,:,:].unsqueeze(-1)
        xyzs2, matches, labels=find_gaussian_peaks(pos_dist2, box_borders, match_threshold=0.80, std=0.20, method='msd_norm')
        bond_coord_prid= list_tensor2array_group(xyzs2)[:, [0,1,2,3]]
        np.savetxt(args.savedir+"bond_coord.txt", bond_coord_prid, fmt="%.6f",delimiter="\t")      # 制表符分隔
        
        plot_pos(atom_coord_prid, args.savedir+"atom_coord.svg", size = 0.2)
        plot_grouppos_line(atom_coord_prid, bond_coord_prid,args.savedir+"bond_coord.svg", size = 0.2)
    
        graph, absent, score = connect5(atom_coord_prid, bond_coord_prid, out=3, use_group = True)
    
        graph_list = [graph[i] for i in range(graph.shape[0])]
        gen_sdfv3000_withoutsmiles(atom_coord_prid, graph_list, score, 'smiles',999, args.savedir+"geometry.sdf")         



if __name__ == "__main__":
    main()
    
    
    
    