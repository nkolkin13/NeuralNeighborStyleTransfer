import torch

def whiten(x, ui, u, s):
    '''
    Applies whitening as described in:
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Chiu_Understanding_Generalized_Whitening_and_Coloring_Transform_for_Universal_Style_Transfer_ICCV_2019_paper.pdf
    x -- N x D pytorch tensor
    ui -- D x D transposed eigenvectors of whitening covariance
    u  -- D x D eigenvectors of whitening covariance
    s  -- D x 1 eigenvalues of whitening covariance
    '''
    tps = lambda x: x.transpose(1, 0)
    return tps(torch.matmul(u, torch.matmul(ui, tps(x)) / s))

def colorize(x, ui, u, s):
    '''
    Applies "coloring transform" as described in:
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Chiu_Understanding_Generalized_Whitening_and_Coloring_Transform_for_Universal_Style_Transfer_ICCV_2019_paper.pdf
    x -- N x D pytorch tensor
    ui -- D x D transposed eigenvectors of coloring covariance
    u  -- D x D eigenvectors of coloring covariance
    s  -- D x 1 eigenvalues of coloring covariance
    '''
    tps = lambda x: x.transpose(1, 0)
    return tps(torch.matmul(u, torch.matmul(ui, tps(x)) * s))

def zca(content, style):
    '''
    Matches the mean and covariance of 'content' to those of 'style'
    content -- N x D pytorch tensor of content feature vectors
    style   -- N x D pytorch tensor of style feature vectors
    '''
    mu_c = content.mean(0, keepdim=True)
    mu_s = style.mean(0, keepdim=True)

    content = content - mu_c
    style = style - mu_s

    cov_c = torch.matmul(content.transpose(1,0), content) / float(content.size(0))
    cov_s = torch.matmul(style.transpose(1,0), style) / float(style.size(0))

    u_c, sig_c, _ = torch.svd(cov_c + torch.eye(cov_c.size(0)).cuda()*1e-4)
    u_s, sig_s, _ = torch.svd(cov_s + torch.eye(cov_s.size(0)).cuda()*1e-4)

    sig_c = sig_c.unsqueeze(1)
    sig_s = sig_s.unsqueeze(1)


    u_c_i = u_c.transpose(1,0)
    u_s_i = u_s.transpose(1,0)

    scl_c = torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8))
    scl_s = torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8))


    whiten_c = whiten(content, u_c_i, u_c, scl_c)
    color_c = colorize(whiten_c, u_s_i, u_s, scl_s) + mu_s

    return color_c, cov_s

def zca_tensor(content, style):
    '''
    Matches the mean and covariance of 'content' to those of 'style'
    content -- B x D x H x W pytorch tensor of content feature vectors
    style   -- B x D x H x W pytorch tensor of style feature vectors
    '''
    content_rs = content.permute(0,2,3,1).contiguous().view(-1,content.size(1))
    style_rs = style.permute(0,2,3,1).contiguous().view(-1,style.size(1))

    cs, cov_s = zca(content_rs, style_rs)

    cs = cs.view(content.size(0),content.size(2),content.size(3),content.size(1)).permute(0,3,1,2)
    return cs.contiguous(), cov_s
