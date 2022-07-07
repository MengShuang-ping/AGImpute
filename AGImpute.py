import argparse
from StatisticalLearningModule import *
from DeepLearningModule import *
import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import gc
from scipy.special import gamma
from sklearn.cluster import KMeans
from torch import nn
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--GPU', action='store_true', default=False, help='Use GPU train')
parser.add_argument('--batch_size', type=int, default=4, help='Size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='Adam: learning rate')
parser.add_argument('--K', type=float, default=0, help='K parameters')
parser.add_argument('--channels', type=int, default=1, help='Number of image channels')
parser.add_argument('--img_size', type=int, default=100, help='Training set size')
parser.add_argument('--epochs_a', type=int, default=200, help='Number of epochs of training autoencoder')
parser.add_argument('--epochs_g', type=int, default=100, help='Number of epochs of training gan')
parser.add_argument('--b1', type=float, default=0.5, help='Adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='Adam: decay of first order momentum of gradient')
parser.add_argument('--gamma', type=float, default=0.95, help='Gamma parameters')
parser.add_argument('--D_throd', type=int, default=20,
                    help='Dropout events threshold positioning algorithm confidence value')
parser.add_argument('--feature_gene_num', type=int, default=4000, help='Number selection of Feature Genes')
parser.add_argument('--dim_thord', type=int, default=1, help='Latent-dim throd')
parser.add_argument('--name', type=str, default='null', help='Name of h5ad')
parser.add_argument('--file_c', type=str, default='.', help='Path of h5ad file')
parser.add_argument('--file_model', type=str, default='.', help='Path of model file')
parser.add_argument('--outdir', type=str, default=".", help='The directory for output.')
opt = parser.parse_args()


def subclustering(data, clu_num):
    kmean = KMeans(n_clusters=clu_num).fit(data)
    cluster_label = kmean.labels_
    return cluster_label


def dropoutenventcal(data, pre_clu):
    data_t = np.asarray(data)  # g*c
    np.seterr(divide='ignore', invalid='ignore')
    m_t, n_t = data_t.shape
    su = (data_t == 0).sum()
    fai = su / m_t / n_t
    nu = n_t * 0.2
    k = nu / (nu + (1 - fai) * n_t)
    ar_data = np.asanyarray(data)
    zinbparm = zinbem(ar_data, fai, k)
    inifai = zinbparm[0]
    inik = zinbparm[2]
    D_throd = opt.D_throd
    dropoutevent = distinguishdropoutevent(data, inifai, inik, pre_clu, D_throd)
    return dropoutevent


def scanpy_proce_variabel_gene(data, gene_num):  # cell c*g*LABEL
    obs = pd.DataFrame()
    obs['index'] = data.index
    var = pd.DataFrame(index=data.columns)
    X = data.values
    adata = ad.AnnData(X, obs=obs, var=var, dtype='float32')
    #     sc.pp.filter_cells(adata, min_genes=200)
    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    out = pd.DataFrame(adata.X)
    out.index = adata.obs.iloc[0:, 0]
    out.columns = adata.var.index
    sc.pp.neighbors(adata, n_neighbors=5, n_pcs=40)
    sc.tl.leiden(adata)
    sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=gene_num)
    variabel_gene = pd.DataFrame(adata.var[adata.var['highly_variable']].index)
    leiden_label = adata.obs.iloc[:, -1]
    return out, variabel_gene, leiden_label


if opt.GPU:
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    Autoencoder_models = opt.file_model + '/Autoencoder_models'  # 设置目录
    if os.path.isdir(Autoencoder_models) != True:  # 是否存在目录，否之创建
        os.makedirs(Autoencoder_models)
    GANs_models = opt.file_model + '/GAN_models'  # 设置目录
    if os.path.isdir(GANs_models) != True:  # 是否存在目录，否之创建
        os.makedirs(GANs_models)


    def train_autoencoder(data, clu_label_ar, img_size):
        label_num = clu_label_ar
        li_data = data.copy()  # need c*g
        li_data['label'] = label_num  # 添加标签列
        dim_size = img_size * img_size
        loss_fn = nn.MSELoss()
        for i in np.unique(label_num):
            print('i', i)
            label_data_li = li_data[li_data['label'] == i].iloc[:, :-1]  # 标签n数据
            running_loss = 0

            automodel = AutoEncoder(label_data_li.shape[1], dim_size)  # 生成对象
            if torch.cuda.is_available():
                automodel.cuda()
                print("autoencoder runing on GPUs")
            else:
                print("autoencoder runing on CPUs")

            #         autoencoder
            automodel.apply(weights_init_normal)
            optimizer_automodel = torch.optim.Adam(automodel.parameters(), lr=lr, betas=(b1, b2))  # 优化
            data_tensor = torch.from_numpy(label_data_li.values).to(torch.float32).cuda()
            # 对象保存名
            auto_model_basename = "auencoder-" + str(opt.name) + "-" + str(i)  # gan+数据名+标签名
            model_exists = os.path.isfile(Autoencoder_models + '/' + auto_model_basename + '-autoencoder.pt')
            if model_exists:
                print("model is exists")
            #         print("The model is exists")
            #         overwrite = "y"
            #         if overwrite != "y":
            #             print("The training was deprecated since optical model exists.")
            #             print("scIGANs continues imputation using existing model...")
            #             sys.exit()  # if model exists and do not want to train again, exit the program
            else:
                cur_loss = 0
                loop = 0
                print("bigan train")
                for epoch in range(opt.epochs_a):

                    add_noise = (label_data_li.values + np.random.poisson(lam=1.0, size=(
                        label_data_li.shape[0], label_data_li.shape[1])))

                    data_tensor_noise = torch.from_numpy(add_noise).to(torch.float32).cuda()

                    data_tensor, data_tensor_noise = Variable(data_tensor).cuda(), Variable(data_tensor_noise).cuda()
                    optimizer_automodel.zero_grad()
                    train_pre = automodel(data_tensor_noise)
                    loss = loss_fn(train_pre[1], data_tensor).cuda()
                    print("AUTOENCODER_LOSS", loss)
                    loss.backward()
                    optimizer_automodel.step()
                    # optimizer.stop()
                    running_loss += loss.item()
                    cur_loss = loss
                    if cur_loss > loss:
                        cur_loss = loss
                        torch.save(automodel.state_dict(),
                                   Autoencoder_models + '/' + auto_model_basename + '-autoencoder.pt')
                        loop += 1
                if loop == 0:
                    torch.save(automodel.state_dict(),
                               Autoencoder_models + '/' + auto_model_basename + '-autoencoder.pt')
                print("finish train")
            print(
                "The optimal model will be output in \"" + os.getcwd() + "/" + Autoencoder_models + "\" with basename = " + auto_model_basename)
        print("finish save model")
        gc.collect()
        torch.cuda.empty_cache()


    def train_gan(data, clu_label_ar, input_img_size):

        k = opt.K
        max_ncls = len(np.unique(clu_label_ar))  # 聚类数
        channels = opt.channels  # 图像通道数
        max_M = sys.float_info.max
        min_dM = 0.001
        dM = 1
        dim_thord = opt.dim_thord
        dim_size = opt.img_size * opt.img_size
        img_size = input_img_size
        latent_dim = int((input_img_size * 2) ** 0.5 + dim_thord)
        out_result_pd = pd.DataFrame()

        generator = Generator(img_size, latent_dim, max_ncls, channels)
        discriminator = Discriminator(img_size, channels, max_ncls)
        if torch.cuda.is_available():
            generator.cuda()
            discriminator.cuda()
            print("gan runing on GPUs")
        else:
            print("gan is runing on CPUs")

        label_num = clu_label_ar
        li_data = data.copy()  # need c*g
        li_data['label'] = label_num  # 添加标签列

        # 返回path最后的文件名
        for i in np.unique(label_num):
            print('i', i)
            loop = 0
            label_data_li = li_data[li_data['label'] == i].iloc[:, :-1]  # 标签n数据
            label_li = li_data[li_data['label'] == i].iloc[:, -1]  # 标签n标签
            dim_size = img_size * img_size  # 自编码器降为多少维

            model_basename = "gan-" + str(opt.name) + "-" + str(i)  # gan+数据名+标签名
            model_exists = os.path.isfile(GANs_models + '/' + model_basename + '-g.pt')
            if model_exists:
                print("The model " + i + " is exists")
            else:
                # 加载auencoder 模型
                auto_model_basename = "auencoder-" + str(opt.name) + "-" + str(i)
                model_auto = Autoencoder_models + '/' + auto_model_basename + '-autoencoder.pt'  # 读autoencoder模型
                automodel = AutoEncoder(label_data_li.shape[1], dim_size)  # 生成对象
                if cuda == True:
                    # discriminator.load_state_dict(torch.load(opt.dpt))
                    automodel.load_state_dict(torch.load(model_auto))
                else:
                    # discriminator.load_state_dict(torch.load(opt.dpt, map_location=lambda storage, loc: storage))
                    automodel.load_state_dict(torch.load(model_auto, map_location=lambda storage, loc: storage))
                    # 这里需要输入自编码的值
                # gan
                # label_data_li_t = label_data_li.T  # need gene*cell
                label_data_li_num = label_data_li.values
                label_data_li_tensor = torch.from_numpy(label_data_li_num).to(torch.float32)  # .cuda()
                label_data_li_encoder = automodel.encoder(label_data_li_tensor)  # 使用自编码降维
                label_data_li_encoder_num = label_data_li_encoder.detach().cpu().numpy()
                label_data_li_encoder_pd = pd.DataFrame(label_data_li_encoder_num)  # 转pd
                transformed_dataset = MyDataset(data=label_data_li_encoder_pd,
                                                label=label_li,
                                                img_size=img_size,
                                                transform=transforms.Compose([
                                                    #                                               Rescale(256),
                                                    #                                               RandomCrop(224),
                                                    ToTensor()
                                                ]))
                dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=0, drop_last=True)

                generator.apply(weights_init_normal)
                discriminator.apply(weights_init_normal)
                # Optimizers
                optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
                optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
                print("begin train gan")

                for epoch in range(opt.epochs_g):
                    cur_M = 0
                    cur_dM = 1
                    for i, batch_sample in enumerate(dataloader):
                        #            if i==0:
                        #                break
                        imgs = batch_sample['data'].type(Tensor)
                        batch_label = batch_sample['label']
                        label_oh = one_hot((batch_label).type(torch.LongTensor), max_ncls).type(Tensor)  #

                        # Configure input
                        real_imgs = Variable(imgs.type(Tensor))

                        optimizer_G.zero_grad()

                        # Sample noise as generator input
                        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

                        # Generate a batch of images
                        gen_imgs = generator(z, label_oh)

                        # Loss measures generator's ability to fool the discriminator
                        g_loss = torch.mean(torch.abs(discriminator(gen_imgs, label_oh) - gen_imgs))

                        g_loss.backward()
                        optimizer_G.step()

                        # ---------------------
                        #  Train Discriminator
                        # ---------------------

                        optimizer_D.zero_grad()

                        # Measure discriminator's ability to classify real from generated samples
                        d_real = discriminator(real_imgs, label_oh)
                        d_fake = discriminator(gen_imgs.detach(), label_oh)

                        d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
                        d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
                        d_loss = d_loss_real - k * d_loss_fake

                        d_loss.backward()
                        optimizer_D.step()

                        # ----------------
                        # Update weights
                        # ----------------

                        diff = torch.mean(gamma * d_loss_real - d_loss_fake)

                        # Update weight term for fake samples
                        k = k + lambda_k * (diff.detach().data.cpu().numpy()).item()
                        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

                        # Update convergence metric
                        M = (d_loss_real + torch.abs(diff)).item()
                        cur_M += M
                        # --------------
                        # Log Progress
                        # --------------

                        sys.stdout.write(
                            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, delta_M: %f,k: %f" % (
                                epoch + 1, opt.epochs_g, i + 1, len(dataloader),
                                (d_loss.detach().data.cpu().numpy()).item(),
                                (g_loss.detach().data.cpu().numpy()).item(),
                                M, dM, k))
                        sys.stdout.flush()

                    cur_M = cur_M / len(dataloader)
                    if cur_M < max_M:  # if current model is better than previous one
                        torch.save(discriminator.state_dict(), GANs_models + '/' + model_basename + '-d.pt')
                        torch.save(generator.state_dict(), GANs_models + '/' + model_basename + '-g.pt')
                        dM = min(max_M - cur_M, cur_M)
                        if dM < min_dM:  # if convergence threthold meets, stop training
                            print(
                                "Training was stopped after " + str(
                                    epoch + 1) + " epoches since the convergence threthold (" + str(
                                    min_dM) + ".) reached: " + str(dM))
                            break
                        cur_dM = max_M - cur_M
                        max_M = cur_M
                        loop += 1
                    if epoch + 1 == opt.epochs_g and cur_dM > loop == 0:  # 最后一次循环保存
                        print("#############################################################")
                        print("最后一次循环保存")
                        torch.save(discriminator.state_dict(), GANs_models + '/' + model_basename + '-d.pt')
                        torch.save(generator.state_dict(), GANs_models + '/' + model_basename + '-g.pt')
                        print("Training was stopped after " + str(
                            epoch + 1) + " epoches since the maximum epoches reached: " + str(
                            opt.epochs_g) + ".")
                        print("WARNING: the convergence threthold (" + str(
                            min_dM) + ") was not met. Current value is: " + str(
                            cur_dM))
                        print("You may need more epoches to get the most optimal model!!!")
                        loop += 1

                    if loop == 0:
                        print("无数据")
                        torch.save(discriminator.state_dict(), GANs_models + '/' + model_basename + '-d.pt')
                        torch.save(generator.state_dict(), GANs_models + '/' + model_basename + '-g.pt')
                gc.collect()
                torch.cuda.empty_cache()
        #   加载模型生成图片

        for i in np.unique(label_num):
            result = []  # 输出生成对应标签的值
            label_data_li = li_data[li_data['label'] == i].iloc[:, :-1]  # 标签n数据
            label_li = li_data[li_data['label'] == i].iloc[:, -1]  # 标签n标签
            sim_size = label_data_li.shape[0]
            # 还要对应标签读取模型 autoencoder 和 gan
            # autoencoder
            auto_model_basename = "auencoder-" + str(opt.name) + "-" + str(i)  # gan+数据名+标签名
            model_auto = Autoencoder_models + '/' + auto_model_basename + '-autoencoder.pt'  # 读autoencoder模型
            automodel = AutoEncoder(label_data_li.shape[1], dim_size)  # 生成对象
            automodel.apply(weights_init_normal)
            if cuda == True:
                # discriminator.load_state_dict(torch.load(opt.dpt))
                automodel.load_state_dict(torch.load(model_auto))
            else:
                # discriminator.load_state_dict(torch.load(opt.dpt, map_location=lambda storage, loc: storage))
                automodel.load_state_dict(torch.load(model_auto, map_location=lambda storage, loc: storage))
            # gan
            generator.apply(weights_init_normal)
            model_basename = "gan-" + str(opt.name) + "-" + str(i)  # gan+数据名+标签名
            model_g = GANs_models + '/' + model_basename + '-g.pt'  # 读模型
            if cuda == True:
                # discriminator.load_state_dict(torch.load(opt.dpt))
                generator.load_state_dict(torch.load(model_g))
            else:
                # discriminator.load_state_dict(torch.load(opt.dpt, map_location=lambda storage, loc: storage))
                generator.load_state_dict(torch.load(model_g, map_location=lambda storage, loc: storage))

            label_oh = one_hot(torch.from_numpy(np.repeat(i, sim_size)).type(torch.LongTensor), max_ncls).type(Tensor)
            # Sample noisae as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (sim_size, latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z, label_oh).detach().data.numpy()
            # 每一个批次求均值叠加到类别再求均值
            print('##############################fake_img', fake_imgs.shape)
            for i in range(fake_imgs.shape[0]):
                for j in range(fake_imgs.shape[1]):
                    li = []
                    for k in range(fake_imgs.shape[2]):
                        li = np.concatenate((li, fake_imgs[i][j][k]), axis=0)
                    result.append(li)
            # 自编码解码
            tensor_pd_result = torch.from_numpy(np.asarray(result)).to(torch.float32)
            print("tensor_pd_resulttensor_pd_resulttensor_pd_resulttensor_pd_result", tensor_pd_result.shape)
            de_result = automodel.decoder(tensor_pd_result).cuda()
            de_result_num = de_result.detach().numpy()
            de_pd_result = pd.DataFrame(de_result_num, index=label_data_li.index,
                                        columns=label_data_li.columns)  # 生成矩阵转pd
            print("de_pd_resultde_pd_resultde_pd_resultde_pd_resultde_pd_resultddddddddddd", de_pd_result.shape)
            de_pd_result_li = de_pd_result.copy()
            # subcluster
            if sim_size > 400:
                print("111111111111111")
                if sim_size <= 400:
                    clu_num = 2
                else:
                    clu_num = int(len(label_li) / 250)
                labels = subclustering(de_result_num, clu_num)
                de_pd_result_li['label'] = labels
                result_mean = []  # 保存均值
                for i in np.unique(labels):
                    clu_li_label_data = de_pd_result_li[de_pd_result_li.iloc[:, -1] == i].iloc[:, :-1]  # 获取当前标签数据
                    # 计算均值
                    li_mean = []
                    for i in range(len(clu_li_label_data.columns)):
                        li_mean.append(clu_li_label_data.iloc[:, i].values.mean())
                    result_mean.append(li_mean)  # 保存均值
                result_mean = np.asarray(result_mean)
                result_mean_pd = pd.DataFrame(result_mean)
                result_mean_pd['label'] = np.unique(labels)
                out_result_li = []
                for i in range(sim_size):  # 循环添加
                    lab = de_pd_result_li.iloc[i, -1]
                    mean_value = result_mean_pd[result_mean_pd.iloc[:, -1] == lab].iloc[:, :-1].values
                    out_result_li.extend(mean_value)
                out_result_li = np.asarray(out_result_li)
                de_out_result = pd.DataFrame(out_result_li, index=label_data_li.index, columns=label_data_li.columns)
            else:
                # 计算均值
                out_result = []
                li_mean = []
                for i in range(len(de_pd_result.columns)):
                    li_mean.append(de_pd_result.iloc[:, i].values.mean())
                # 保存均值
                for i in label_li:
                    mean_value = li_mean
                    out_result.append(mean_value)
                out_result = np.asarray(out_result)

                de_out_result = pd.DataFrame(out_result, index=label_data_li.index, columns=label_data_li.columns)
            out_result_pd = pd.concat([out_result_pd, de_out_result], axis=0)
        return out_result_pd


    if __name__ == '__main__':
        # 超参数
        batch_size = opt.batch_size
        lr = opt.lr
        weight_decay = 1e-5
        b1 = opt.b1
        b2 = opt.b2
        running_loss = 0  # 图片大小
        a = 1
        lambda_k = 0.001
        gamma = opt.gamma
        gc.collect()
        torch.cuda.empty_cache()
        raw_data = pd.read_csv(opt.file_c + opt.name + '.csv', index_col=0)
        proce_data, variable_gene, leiden_label = scanpy_proce_variabel_gene(raw_data, opt.feature_gene_num)

        data = proce_data
        feature_gene = []  # 标记基因的位置
        for i in range(data.shape[1]):
            if data.columns[i] in variable_gene.values:
                feature_gene.append(i)

        # 获取预聚类指标
        label_name = "pre_leida_clu-" + opt.name
        label_exists = os.path.isfile(opt.file_model + label_name + '.csv')
        if label_exists:
            clu_label_ar = np.asarray(
                pd.read_csv(opt.file_model + label_name + '.csv', index_col=0).iloc[:, 0])
        else:
            clu_label_ar = np.asarray(leiden_label.astype(np.int32))
            pd.DataFrame(clu_label_ar).to_csv(opt.file_model + label_name + '.csv')

        li_data = data.copy()
        label_data = li_data.copy()
        label_data['label'] = clu_label_ar

        train_autoencoder(data, clu_label_ar, opt.img_size)
        out_result = train_gan(data, clu_label_ar, opt.img_size)
        dropoutevent = dropoutenventcal(data, clu_label_ar)
        num_g = 0
        num_i = 0
        for i in dropoutevent:
            if i[1] in feature_gene:
                num_g += 1
                target_label = clu_label_ar[i[0]]
                cut_data = label_data[label_data.iloc[:, -1] == target_label]
                por_zero = sum(cut_data.iloc[:, i[0]] == 0) / cut_data.shape[0]
                if por_zero < 0.95:  # 共表达
                    value = out_result.loc[li_data.index[i[0]], li_data.columns[i[1]]]
                    li_data.iloc[i[0], i[1]] = value
                    num_i += 1

            print(num_g)

        impute_data = li_data  # 插补数据
        out_dir = opt.outdir + '/result'
        if os.path.isdir(out_dir) != True:  # 是否存在目录，否之创建
            os.makedirs(out_dir)
        impute_data.to_csv(opt.outdir + "/result/agimpute_{}.csv".format(opt.name))
        print("ALL FINISH")


else:
    Tensor = torch.FloatTensor
    # 遍历标签
    Autoencoder_models = opt.file_model + '/Autoencoder_models'  # 设置目录
    if os.path.isdir(Autoencoder_models) != True:  # 是否存在目录，否之创建
        os.makedirs(Autoencoder_models)
    GANs_models = opt.file_model + '/GAN_models'  # 设置目录
    if os.path.isdir(GANs_models) != True:  # 是否存在目录，否之创建
        os.makedirs(GANs_models)


    def train_autoencoder(data, clu_label_ar, img_size):
        label_num = clu_label_ar
        li_data = data.copy()  # need c*g
        li_data['label'] = label_num  # 添加标签列
        dim_size = img_size * img_size
        loss_fn = nn.MSELoss()
        for i in np.unique(label_num):
            print('i', i)
            label_data_li = li_data[li_data['label'] == i].iloc[:, :-1]  # 标签n数据
            label_li = li_data[li_data['label'] == i].iloc[:, -1]  # 标签n标签
            label_li_len = len(np.unique(label_li))
            running_loss = 0

            automodel = AutoEncoder(label_data_li.shape[1], dim_size)  # 生成对象

            #         autoencoder
            automodel.apply(weights_init_normal)
            optimizer_automodel = torch.optim.Adam(automodel.parameters(), lr=lr, betas=(b1, b2))  # 优化
            data_tensor = torch.from_numpy(label_data_li.values).to(torch.float32)
            # 对象保存名
            auto_model_basename = "auencoder-" + str(opt.name) + "-" + str(i)  # gan+数据名+标签名
            model_exists = os.path.isfile(Autoencoder_models + '/' + auto_model_basename + '-autoencoder.pt')
            if model_exists:
                print("model is exists")
            else:
                cur_loss = 0
                loop = 0
                print("bigan train")
                for epoch in range(opt.epochs_a):

                    add_noise = (label_data_li.values + np.random.poisson(lam=1.0, size=(
                        label_data_li.shape[0], label_data_li.shape[1])))

                    data_tensor_noise = torch.from_numpy(add_noise).to(torch.float32)

                    data_tensor, data_tensor_noise = Variable(data_tensor), Variable(data_tensor_noise)
                    optimizer_automodel.zero_grad()
                    train_pre = automodel(data_tensor_noise)
                    loss = loss_fn(train_pre[1], data_tensor)
                    print("AUTOENCODER_LOSS", loss)
                    loss.backward()
                    optimizer_automodel.step()
                    # optimizer.stop()
                    running_loss += loss.item()
                    cur_loss = loss
                    if cur_loss > loss:
                        cur_loss = loss
                        torch.save(automodel.state_dict(),
                                   Autoencoder_models + '/' + auto_model_basename + '-autoencoder.pt')
                        loop += 1
                if loop == 0:
                    torch.save(automodel.state_dict(),
                               Autoencoder_models + '/' + auto_model_basename + '-autoencoder.pt')
                print("finish train")
            print(
                "The optimal model will be output in \"" + os.getcwd() + "/" + Autoencoder_models + "\" with basename = " + auto_model_basename)
        print("finish save model")
        gc.collect()
        torch.cuda.empty_cache()


    ## 神经网络模块

    def train_gan(data, clu_label_ar, input_img_size):
        k = opt.K
        max_ncls = len(np.unique(clu_label_ar))  # 聚类数
        channels = opt.channels  # 图像通道数
        max_M = sys.float_info.max
        min_dM = 0.001
        dM = 1
        dim_thord = opt.dim_thord
        img_size = input_img_size
        latent_dim = int((input_img_size * 2) ** 0.5 + dim_thord)
        out_result_pd = pd.DataFrame()

        generator = Generator(img_size, latent_dim, max_ncls, channels)
        discriminator = Discriminator(img_size, channels, max_ncls)

        loss_fn = nn.MSELoss()

        label_num = clu_label_ar
        li_data = data.copy()  # need c*g
        li_data['label'] = label_num  # 添加标签列

        # 返回path最后的文件名
        for i in np.unique(label_num):
            print('i', i)
            loop = 0
            label_data_li = li_data[li_data['label'] == i].iloc[:, :-1]  # 标签n数据
            label_li = li_data[li_data['label'] == i].iloc[:, -1]  # 标签n标签
            dim_size = opt.img_size * opt.img_size  # 自编码器降为多少维

            model_basename = "gan-" + str(opt.name) + "-" + str(i)  # gan+数据名+标签名
            model_exists = os.path.isfile(GANs_models + '/' + model_basename + '-g.pt')
            if model_exists:
                print("The model is exists")
            else:
                # 加载auencoder 模型
                auto_model_basename = "auencoder-" + str(opt.name) + "-" + str(i)
                model_auto = Autoencoder_models + '/' + auto_model_basename + '-autoencoder.pt'  # 读autoencoder模型
                automodel = AutoEncoder(label_data_li.shape[1], dim_size)  # 生成对象
                automodel.load_state_dict(torch.load(model_auto, map_location=lambda storage, loc: storage))
                # 这里需要输入自编码的值
                # gan
                label_data_li_num = label_data_li.values
                label_data_li_tensor = torch.from_numpy(label_data_li_num).to(torch.float32)  #
                label_data_li_encoder = automodel.encoder(label_data_li_tensor)  # 使用自编码降维
                label_data_li_encoder_num = label_data_li_encoder.detach().numpy()
                label_data_li_encoder_pd = pd.DataFrame(label_data_li_encoder_num)  # 转pd
                transformed_dataset = MyDataset(data=label_data_li_encoder_pd,
                                                label=label_li,
                                                img_size=img_size,
                                                transform=transforms.Compose([
                                                    ToTensor()
                                                ]))
                dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=0, drop_last=True)

                generator.apply(weights_init_normal)
                discriminator.apply(weights_init_normal)
                # Optimizers
                optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
                optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
                print("begin train gan")

                for epoch in range(opt.epochs_g):
                    cur_M = 0
                    cur_dM = 1
                    for i, batch_sample in enumerate(dataloader):
                        imgs = batch_sample['data'].type(Tensor)
                        batch_label = batch_sample['label']
                        label_oh = one_hot((batch_label).type(torch.LongTensor), max_ncls).type(Tensor)  #

                        real_imgs = Variable(imgs.type(Tensor))

                        optimizer_G.zero_grad()

                        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

                        # Generate a batch of images
                        gen_imgs = generator(z, label_oh)

                        # Loss measures generator's ability to fool the discriminator
                        g_loss = torch.mean(torch.abs(discriminator(gen_imgs, label_oh) - gen_imgs))

                        g_loss.backward()
                        optimizer_G.step()

                        optimizer_D.zero_grad()
                        d_real = discriminator(real_imgs, label_oh)
                        d_fake = discriminator(gen_imgs.detach(), label_oh)

                        d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
                        d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
                        d_loss = d_loss_real - k * d_loss_fake

                        d_loss.backward()
                        optimizer_D.step()

                        diff = torch.mean(gamma * d_loss_real - d_loss_fake)

                        # Update weight term for fake samples
                        k = k + lambda_k * (diff.detach().data.numpy()).item()
                        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

                        # Update convergence metric
                        M = (d_loss_real + torch.abs(diff)).item()
                        cur_M += M

                        sys.stdout.write(
                            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, delta_M: %f,k: %f" % (
                                epoch + 1, opt.epochs_g, i + 1, len(dataloader),
                                (d_loss.detach().data.numpy()).item(),
                                (g_loss.detach().data.numpy()).item(),
                                M, dM, k))
                        sys.stdout.flush()
                        # get the M of current epoch

                    cur_M = cur_M / len(dataloader)
                    if cur_M < max_M:  # if current model is better than previous one
                        torch.save(discriminator.state_dict(), GANs_models + '/' + model_basename + '-d.pt')
                        torch.save(generator.state_dict(), GANs_models + '/' + model_basename + '-g.pt')
                        dM = min(max_M - cur_M, cur_M)
                        if dM < min_dM:  # if convergence threthold meets, stop training
                            print(
                                "Training was stopped after " + str(
                                    epoch + 1) + " epoches since the convergence threthold (" + str(
                                    min_dM) + ".) reached: " + str(dM))
                            break
                        cur_dM = max_M - cur_M
                        max_M = cur_M
                        loop += 1

                    if epoch + 1 == opt.epochs_g and cur_dM > loop == 0:  # 最后一次循环保存
                        torch.save(discriminator.state_dict(), GANs_models + '/' + model_basename + '-d.pt')
                        torch.save(generator.state_dict(), GANs_models + '/' + model_basename + '-g.pt')
                        print("You may need more epoches to get the most optimal model!!!")
                        loop += 1

                    if loop == 0:
                        print("无数据")
                        torch.save(discriminator.state_dict(), GANs_models + '/' + model_basename + '-d.pt')
                        torch.save(generator.state_dict(), GANs_models + '/' + model_basename + '-g.pt')
                gc.collect()
                torch.cuda.empty_cache()
        # 加载模型生成图片

        for i in np.unique(label_num):
            print("加载模型生成图片")
            print('i', i)
            result = []  # 输出生成对应标签的值
            label_data_li = li_data[li_data['label'] == i].iloc[:, :-1]  # 标签n数据
            label_li = li_data[li_data['label'] == i].iloc[:, -1]  # 标签n标签
            sim_size = label_data_li.shape[0]
            # 还要对应标签读取模型 autoencoder 和 gan
            # autoencoder
            auto_model_basename = "auencoder-" + str(opt.name) + "-" + str(i)  # gan+数据名+标签名
            model_auto = Autoencoder_models + '/' + auto_model_basename + '-autoencoder.pt'  # 读autoencoder模型
            automodel = AutoEncoder(label_data_li.shape[1], dim_size)  # 生成对象
            automodel.apply(weights_init_normal)
            automodel.load_state_dict(torch.load(model_auto, map_location=lambda storage, loc: storage))
            # gan
            generator.apply(weights_init_normal)
            model_basename = "gan-" + str(opt.name) + "-" + str(i)  # gan+数据名+标签名
            model_g = GANs_models + '/' + model_basename + '-g.pt'  # 读模型
            generator.load_state_dict(torch.load(model_g, map_location=lambda storage, loc: storage))

            label_oh = one_hot(torch.from_numpy(np.repeat(i, sim_size)).type(torch.LongTensor), max_ncls).type(Tensor)
            z = Variable(Tensor(np.random.normal(0, 1, (sim_size, latent_dim))))

            fake_imgs = generator(z, label_oh).detach().data.numpy()

            for i in range(fake_imgs.shape[0]):
                for j in range(fake_imgs.shape[1]):
                    li = []
                    for k in range(fake_imgs.shape[2]):
                        li = np.concatenate((li, fake_imgs[i][j][k]), axis=0)
                    result.append(li)

            # 自编码解码
            tensor_pd_result = torch.from_numpy(np.asarray(result)).to(torch.float32)
            de_result = automodel.decoder(tensor_pd_result)
            de_result_num = de_result.detach().numpy()
            de_pd_result = pd.DataFrame(de_result_num, index=label_data_li.index,
                                        columns=label_data_li.columns)  # 生成矩阵转pd
            de_pd_result_li = de_pd_result.copy()
            # 子聚类
            if sim_size > 400:
                if sim_size <= 400:
                    clu_num = 2
                else:
                    clu_num = int(len(label_li) / 250)
                labels = subclustering(de_result_num, clu_num)
                de_pd_result_li['label'] = labels
                result_mean = []  # 保存均值
                for i in np.unique(labels):
                    clu_li_label_data = de_pd_result_li[de_pd_result_li.iloc[:, -1] == i].iloc[:, :-1]  # 获取当前标签数据
                    # 计算均值
                    li_mean = []
                    for i in range(len(clu_li_label_data.columns)):
                        li_mean.append(clu_li_label_data.iloc[:, i].values.mean())
                    result_mean.append(li_mean)  # 保存均值
                result_mean = np.asarray(result_mean)
                result_mean_pd = pd.DataFrame(result_mean)
                result_mean_pd['label'] = np.unique(labels)
                out_result_li = []
                for i in range(sim_size):  # 循环添加
                    lab = de_pd_result_li.iloc[i, -1]
                    mean_value = result_mean_pd[result_mean_pd.iloc[:, -1] == lab].iloc[:, :-1].values
                    out_result_li.extend(mean_value)
                out_result_li = np.asarray(out_result_li)
                de_out_result = pd.DataFrame(out_result_li, index=label_data_li.index, columns=label_data_li.columns)
            else:
                # 计算均值
                out_result = []
                li_mean = []
                for i in range(len(de_pd_result.columns)):
                    li_mean.append(de_pd_result.iloc[:, i].values.mean())
                # 保存均值
                for i in label_li:
                    mean_value = li_mean
                    out_result.append(mean_value)
                out_result = np.asarray(out_result)

                de_out_result = pd.DataFrame(out_result, index=label_data_li.index, columns=label_data_li.columns)
            out_result_pd = pd.concat([out_result_pd, de_out_result], axis=0)
        return out_result_pd


    if __name__ == '__main__':
        # 超参数
        batch_size = opt.batch_size
        lr = opt.lr
        weight_decay = 1e-5
        b1 = opt.b1
        b2 = opt.b2
        running_loss = 0  # 图片大小
        a = 1
        lambda_k = 0.001
        gamma = opt.gamma
        gc.collect()
        torch.cuda.empty_cache()
        # TODO

        raw_data = pd.read_csv(opt.file_c + opt.name + '.csv', index_col=0)
        proce_data, variable_gene, leiden_label = scanpy_proce_variabel_gene(raw_data, opt.feature_gene_num)

        data = proce_data
        feature_gene = []  # 标记基因的位置
        for i in range(data.shape[1]):
            if data.columns[i] in variable_gene.values:
                feature_gene.append(i)

        # 获取预聚类指标
        label_name = "pre_leida_clu-" + opt.name
        label_exists = os.path.isfile(opt.file_model + label_name + '.csv')
        if label_exists:
            clu_label_ar = np.asarray(
                pd.read_csv(opt.file_model + label_name + '.csv', index_col=0).iloc[:, 0])
        else:
            clu_label_ar = np.asarray(leiden_label.astype(np.int32))
            pd.DataFrame(clu_label_ar).to_csv(opt.file_model + label_name + '.csv')

        li_data = data.copy()
        label_data = li_data.copy()
        label_data['label'] = clu_label_ar

        train_autoencoder(data, clu_label_ar, opt.img_size)
        out_result = train_gan(data, clu_label_ar, opt.img_size)
        print(out_result)
        num_g = 0
        num_i = 0
        dropoutevent = dropoutenventcal(data, clu_label_ar)
        for i in dropoutevent:
            if i[1] in feature_gene:
                num_g += 1
                target_label = clu_label_ar[i[0]]
                cut_data = label_data[label_data.iloc[:, -1] == target_label]
                por_zero = sum(cut_data.iloc[:, i[0]] == 0) / cut_data.shape[0]
                if por_zero < 0.95:  # 共表达
                    value = out_result.loc[li_data.index[i[0]], li_data.columns[i[1]]]
                    li_data.iloc[i[0], i[1]] = value
                    num_i += 1
        impute_data = li_data  # 插补数据
        out_dir = opt.outdir + '/result'
        if os.path.isdir(out_dir) != True:  # 是否存在目录，否之创建
            os.makedirs(out_dir)
        impute_data.to_csv(opt.outdir + "/result/agimpute_{}.csv".format(opt.name))
        print("ALL FINISH")
