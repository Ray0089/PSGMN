import torch
from torch import nn
from torch.nn import Linear as Lin
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from .resnet import resnet18
from torch_geometric.data import Data
import torch_geometric.transforms as T
import numpy as np
from utils.utils import load_ply,read_ply_to_data
from csrc.node_index_projection import node_project,resize_mask,resize_img

class psgmn(torch.nn.Module):

    def __init__(self,mesh_dir,
                 img_in_channels = 64, mesh_in_channels = 9, out_channels = 64,
                 img_dim=2, mesh_dim=3, num_img_layers=2, num_mesh_layers=6,
                 fcdim=256, s8dim=128, s4dim=64, s2dim=64, raw_dim=64,
                 seg_dim = 2,feature_dim=64,
                 cat=True, lin=True, dropout=0.1,sigma=100):

        super(psgmn,self).__init__()

        self.sigma = sigma
        mesh_model = load_ply(mesh_dir)
        self.mesh_node = torch.tensor(mesh_model['pts'], dtype=torch.float32)
        self.face_idx = torch.tensor(mesh_model['faces'], dtype=torch.int)
        self.mesh_graph = read_ply_to_data(mesh_dir)
        x,edge_idx,edge_attr = self.set_mesh_graph()
        self.resize_ratio_min = 0.8
        self.resize_ratio_max = 1.2

        self.register_buffer('mesh_graph_x',x)
        self.register_buffer('mesh_graph_edge_index',edge_idx)
        self.register_buffer('mesh_graph_edge_attr',edge_attr)
        #self.register_buffer('const_one',torch.tensor(1))
     
        self.img_in_channels = img_in_channels
        self.mesh_in_channels = mesh_in_channels
        self.out_channels = out_channels

        self.num_img_layers = num_img_layers
        self.num_mesh_layers = num_mesh_layers

        self.img_dim = img_dim
        self.mesh_dim = mesh_dim

        self.lin = Lin
        self.cat = cat
        self.dropout = dropout
        self.img_convs = torch.nn.ModuleList()
        self.mesh_convs = torch.nn.ModuleList()
        self.seg_dim = seg_dim
        self.feature_dim = feature_dim
        self.loss_fn = nn.CrossEntropyLoss()
        self.seg_loss = nn.CrossEntropyLoss()

        for _ in range(self.num_mesh_layers):
            mesh_conv = SplineConv(self.mesh_in_channels, out_channels, self.mesh_dim, kernel_size = 5)
            self.mesh_convs.append(mesh_conv)
            self.mesh_in_channels = out_channels

        if self.cat:

            mesh_in_channel = mesh_in_channels + num_mesh_layers * out_channels
        else:

            mesh_in_channel = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.mesh_final = Lin(mesh_in_channel,out_channels)
        else:
            self.out_channels = in_channels

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

         # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )

        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_raw = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, seg_dim+feature_dim, 1, 1)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

    def project(self,K,RT):
        device = K.device
        xyz = torch.matmul(self.mesh_node.to(device), RT[:, :3].t()) + RT[:, 3:].t()
        xyz = torch.matmul(xyz, K.t().float())
        xy = (xyz[:, :2] / xyz[:, 2:]).int()
        z = xyz[:,2].float()
        return xy,z

    def crop_mat(self,img,mask,node_idx,target_h,target_w,overlap_ratio=0.5):

        channels, h, w = img.shape   # 480 640
        hs, ws = torch.where(mask)   
  
        hmin, hmax = torch.min(hs), torch.max(hs)  # mask bbox
        wmin, wmax = torch.min(ws), torch.max(ws)
        fh, fw = hmax - hmin, wmax - wmin  # mask height width
        pad = target_h >= h    # if mask h,w > th,tw  pad the image

        if pad:
            new_img = torch.zeros((channels,target_h,target_w),dtype=torch.float32,device=img.device)
            new_mask = torch.zeros((target_h,target_w),dtype=torch.int,device=img.device)
            new_node_idx = -torch.ones((target_h,target_w),dtype=torch.int,device=img.device)

            new_img[:, :h,:w] = img
            new_mask[:h,:w] = mask
            new_node_idx[:h,:w] = node_idx

            img = resize_img(new_img,h,w)
            mask = resize_mask(new_mask.int(),h,w)
            node_idx = resize_mask(new_node_idx.int(),h,w)
            return img, mask, node_idx

        hrmax = int(min(hmin + overlap_ratio * fh, h - target_h)) # h must > target_height else hrmax<0
        hrmin = int(max(hmin + overlap_ratio * fh - target_h, 0)) 
        wrmax = int(min(wmin + overlap_ratio * fw, w - target_w))   # w must > target_width else wrmax<0
        wrmin = int(max(wmin + overlap_ratio * fw - target_w, 0))

        hbeg = np.random.randint(hrmin,hrmax)
        hend = hbeg + target_h
        wbeg = np.random.randint(wrmin,wrmax) 
        wend = wbeg + target_w

        new_img = img[:, hbeg:hend, wbeg:wend].clone().detach()
        new_mask = mask[hbeg:hend, wbeg:wend].clone().detach()
        new_node_idx = node_idx[hbeg:hend, wbeg:wend].clone().detach()

        img = resize_img(new_img,h,w)
        mask = resize_mask(new_mask.int(),h,w)
        node_idx = resize_mask(new_node_idx.int(),h,w)

        return img, mask, node_idx

    
    def set_mesh_graph(self):
        mesh_transform = T.Compose([
                T.FaceToEdge(),
                T.Cartesian()
            ])
        mesh_graph = mesh_transform(self.mesh_graph)
        return mesh_graph.x, mesh_graph.edge_index,mesh_graph.edge_attr

    def resize_mat(self,img,mask,pose,K):

        device = img.device
        batch_size,_,h,w = img.shape
        correspond_node_idx = - torch.ones((batch_size,h,w),dtype=torch.int,device=device)

        for i in range(batch_size):
            if torch.sum(mask[i]).item() is 0:
                continue
            pts,depth = self.project(K[i],pose[i])
            v,u = torch.where(mask[i])

            node_idx = node_project(u.int(),v.int(),self.face_idx.to(device),pts,depth)
            correspond_node_idx[i][(v,u)] = node_idx
          
            resize_ratio = np.random.uniform(self.resize_ratio_min, self.resize_ratio_max)
            target_height = int(h * resize_ratio)
            target_width = int(w * resize_ratio)
            img[i],mask[i],correspond_node_idx[i] = \
                self.crop_mat(\
                    img[i].clone().detach(),mask[i].clone().detach(),\
                        correspond_node_idx[i].clone().detach(),target_height,target_width)
                  
        return img,mask,correspond_node_idx
        
    def forward(self,x,mask=None,pose=None,K=None,*args):

        if self.training:

            x,mask,node_index = self.resize_mat(\
                x.clone().detach(),mask.clone().detach(),pose,K)

        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)

        fm=self.conv8s(torch.cat([xfc,x8s],1))
        fm=self.up8sto4s(fm)
        if fm.shape[2]==136:
            fm = nn.functional.interpolate(fm, (135,180), mode='bilinear', align_corners=False)

        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm=self.up4sto2s(fm)

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)
        fm=self.conv_raw(torch.cat([fm,x],1))
        
        seg = fm[:,:self.seg_dim,:,:]
        features = fm[:,self.seg_dim:,:,:]
        features = features.permute(0,2,3,1)

        features = features.reshape(-1,features.shape[-1])

        meshes = [self.mesh_graph_x]
        for conv in self.mesh_convs:
            meshes += [F.relu(conv(
                meshes[-1], self.mesh_graph_edge_index, self.mesh_graph_edge_attr),inplace=True)]
            
        # ## TODO:add bn
        
        if self.cat:
            meshes = torch.cat(meshes, dim = -1)
        else:
            meshes = meshes[-1]

        meshes = F.dropout(meshes, p=self.dropout, training=self.training)

        if self.lin:
            meshes = self.mesh_final(meshes)
         
        if self.training:
            idx = torch.where(mask.reshape(-1))[0]
            
        else:
            mask_pred = torch.argmax(seg,dim=1)
            idx = torch.where(mask_pred.reshape(-1))[0]
            
        img_feauture_selected = features.index_select(0,idx).permute(1,0)

        S = torch.matmul(meshes,img_feauture_selected)

        if self.training:

            S = S.unsqueeze(0)
            device = S.device

            mesh_pts_ind = torch.masked_select(node_index,mask.bool())

            valid_pts_ind = torch.where(mesh_pts_ind != torch.tensor(-1,device=device))[0]
            mesh_pts_ind=mesh_pts_ind.unsqueeze(0)

            match_loss_ = self.loss_fn(S[:,:,valid_pts_ind],mesh_pts_ind[:,valid_pts_ind].long())
            seg_loss_ = self.seg_loss(seg,mask.long()) * self.sigma

            loss={}
            loss['seg'] = seg_loss_
            loss['match'] = match_loss_
            return loss

        else:
            S = S.permute(1,0)  #for data parallel
            return S,seg
