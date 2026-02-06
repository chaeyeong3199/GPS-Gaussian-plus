import numpy as np
import torch
from plyfile import PlyData, PlyElement
from gaussian_renderer import render

# [NEW] 저장 함수 추가
def save_ply_3dgs(path, xyz, rgb, opacity, scale, rotation):
    # Tensor -> Numpy 변환
    xyz = xyz.detach().cpu().numpy()
    rgb = rgb.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()
    scale = scale.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()

    # 3DGS Viewer 호환성을 위한 데이터 변환
    # 1. RGB -> SH DC term 변환 (Viewer는 SH를 기대함)
    f_dc = (rgb - 0.5) / 0.28209479177387814

    # 2. Opacity -> Logit (Inverse Sigmoid)
    opacity = np.clip(opacity, 1e-6, 1 - 1e-6)
    opacity = np.log(opacity / (1 - opacity))

    # 3. Scale -> Log Scale
    scale = np.log(np.clip(scale, 1e-8, None))

    # PLY 구조체 생성
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    
    # 45개의 나머지 SH 계수는 0으로 채움 (View dependency 제외)
    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))

    elements = np.zeros(xyz.shape[0], dtype=dtype)
    
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['f_dc_0'], elements['f_dc_1'], elements['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    elements['opacity'] = opacity[:, 0]
    elements['scale_0'], elements['scale_1'], elements['scale_2'] = scale[:, 0], scale[:, 1], scale[:, 2]
    elements['rot_0'], elements['rot_1'], elements['rot_2'], elements['rot_3'] = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"Saved PLY to {path}")

def pts2render(data, bg_color, save_ply_path=None):
    '''
    :param data: rgb input color [-1, 1], will be scaled to [0, 1]
    :param bg_color:  [0, 0, 0]
    :return: rbg render result in [0, 1]
    '''
    bs = data['lmain']['img'].shape[0]

    render_novel_list = []
    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        for view in ['lmain', 'rmain']:
            valid_i = data[view]['pts_valid'][i, :]
            xyz_i = data[view]['xyz'][i, :, :]  # [S*S, 3]
            rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).view(-1, 3)  # [S*S, 3]
            # rgb_i = data[view]['color_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)  # [S*S, 3]
            rot_i = data[view]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)  # [S*S, 4]
            scale_i = data[view]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)  # [S*S, 3]
            opacity_i = data[view]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)  # [S*S, 1]

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3)) #[valid_i]
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
        pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        if save_ply_path is not None:
             # 파일명 중복 방지 (배치 사이즈 > 1 인 경우 고려)
             save_name = save_ply_path if bs == 1 else save_ply_path.replace(".ply", f"_{i}.ply")
             save_ply_3dgs(save_name, pts_xyz_i, pts_rgb_i, opacity_i, scale_i, rot_i)

        render_novel_i = render(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color)
        render_novel_list.append(render_novel_i.unsqueeze(0))

    data['novel_view']['img_pred'] = torch.concat(render_novel_list, dim=0)
    return data
