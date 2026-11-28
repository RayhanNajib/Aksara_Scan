"""
Arsitektur Detektor Objek YOLOX (Apache-2.0 Compatible, Bebas AGPL-3.0) — AksaraScan v3.0
Mendukung varian:
  - yolox-nano  (depth=0.33, width=0.25, depthwise=True)  -> ~3.7 MB FP32 ONNX
  - yolox-tiny  (depth=0.33, width=0.375, depthwise=False) -> ~19 MB FP32 / ~5.3 MB INT8 ONNX
  - yolox-s     (depth=0.33, width=0.50, depthwise=False)  -> ~17 MB FP16 / ~9.2 MB INT8 ONNX
Opsional fitur `use_p2=True` (menambah level P2 stride-4) untuk menjaga recall tinggi pada
tanda aksara Jawa berukuran sangat kecil (< 16 px seperti cecak, wulu, layar, pada_lingsa).

Format keluaran eval/ekspor ONNX: tensor tunggal [B, 4 + nc, num_anchors]
di mana baris 0..3 = (cx, cy, w, h) dalam piksel [0..imgsz], dan baris 4..4+nc = sigmoid(obj) * sigmoid(cls).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class BaseConv(nn.Module):
    def __init__(self, in_c, out_c, ksize, stride=1, groups=1):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, ksize, stride, pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, in_c, out_c, ksize, stride=1):
        super().__init__()
        self.dconv = BaseConv(in_c, in_c, ksize, stride=stride, groups=in_c)
        self.pconv = BaseConv(in_c, out_c, 1, stride=1, groups=1)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, shortcut=True, expansion=0.5, depthwise=False):
        super().__init__()
        hid = int(out_c * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_c, hid, 1)
        self.conv2 = Conv(hid, out_c, 3)
        self.use_add = shortcut and in_c == out_c

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_add else y


class CSPLayer(nn.Module):
    def __init__(self, in_c, out_c, n=1, shortcut=True, expansion=0.5, depthwise=False):
        super().__init__()
        hid = int(out_c * expansion)
        self.conv1 = BaseConv(in_c, hid, 1)
        self.conv2 = BaseConv(in_c, hid, 1)
        self.conv3 = BaseConv(2 * hid, out_c, 1)
        self.m = nn.Sequential(*[Bottleneck(hid, hid, shortcut, 1.0, depthwise) for _ in range(n)])

    def forward(self, x):
        x1 = self.m(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))


class SPPBottleneck(nn.Module):
    def __init__(self, in_c, out_c, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hid = in_c // 2
        self.conv1 = BaseConv(in_c, hid, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernel_sizes])
        self.conv2 = BaseConv(hid * (len(kernel_sizes) + 1), out_c, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class Focus(nn.Module):
    def __init__(self, in_c, out_c, ksize=3):
        super().__init__()
        self.conv = BaseConv(in_c * 4, out_c, ksize)

    def forward(self, x):
        # Slice space-to-depth (kompatibel ONNX opset 12+)
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ], dim=1))


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul=0.33, wid_mul=0.375, depthwise=False):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        base_c = int(wid_mul * 64)
        base_d = max(round(dep_mul * 3), 1)

        self.stem = Focus(3, base_c, 3)  # stride 2
        self.dark2 = nn.Sequential(
            Conv(base_c, base_c * 2, 3, stride=2),
            CSPLayer(base_c * 2, base_c * 2, n=base_d, depthwise=depthwise),
        )  # stride 4 (P2)
        self.dark3 = nn.Sequential(
            Conv(base_c * 2, base_c * 4, 3, stride=2),
            CSPLayer(base_c * 4, base_c * 4, n=base_d * 3, depthwise=depthwise),
        )  # stride 8 (P3)
        self.dark4 = nn.Sequential(
            Conv(base_c * 4, base_c * 8, 3, stride=2),
            CSPLayer(base_c * 8, base_c * 8, n=base_d * 3, depthwise=depthwise),
        )  # stride 16 (P4)
        self.dark5 = nn.Sequential(
            Conv(base_c * 8, base_c * 16, 3, stride=2),
            SPPBottleneck(base_c * 16, base_c * 16),
            CSPLayer(base_c * 16, base_c * 16, n=base_d, shortcut=False, depthwise=depthwise),
        )  # stride 32 (P5)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.dark2(x)
        c3 = self.dark3(c2)
        c4 = self.dark4(c3)
        c5 = self.dark5(c4)
        return c2, c3, c4, c5


class YOLOXPAFPN(nn.Module):
    def __init__(self, dep_mul=0.33, wid_mul=0.375, depthwise=False, use_p2=False):
        super().__init__()
        self.use_p2 = use_p2
        Conv = DWConv if depthwise else BaseConv
        in_c = [int(wid_mul * c) for c in (128, 256, 512, 1024)]  # c2, c3, c4, c5
        n_csp = max(round(dep_mul * 3), 1)

        self.backbone = CSPDarknet(dep_mul, wid_mul, depthwise)

        # Top-down
        self.lateral_conv0 = BaseConv(in_c[3], in_c[2], 1)
        self.C3_p4 = CSPLayer(2 * in_c[2], in_c[2], n=n_csp, shortcut=False, depthwise=depthwise)
        self.reduce_conv1 = BaseConv(in_c[2], in_c[1], 1)
        self.C3_p3 = CSPLayer(2 * in_c[1], in_c[1], n=n_csp, shortcut=False, depthwise=depthwise)

        if self.use_p2:
            self.reduce_conv2 = BaseConv(in_c[1], in_c[0], 1)
            self.C3_p2 = CSPLayer(2 * in_c[0], in_c[0], n=n_csp, shortcut=False, depthwise=depthwise)
            self.bu_conv_p2 = Conv(in_c[0], in_c[0], 3, stride=2)
            self.C3_n2 = CSPLayer(in_c[0] + in_c[1], in_c[1], n=n_csp, shortcut=False, depthwise=depthwise)

        # Bottom-up
        self.bu_conv2 = Conv(in_c[1], in_c[1], 3, stride=2)
        self.C3_n3 = CSPLayer(2 * in_c[1], in_c[2], n=n_csp, shortcut=False, depthwise=depthwise)
        self.bu_conv1 = Conv(in_c[2], in_c[2], 3, stride=2)
        self.C3_n4 = CSPLayer(2 * in_c[2], in_c[3], n=n_csp, shortcut=False, depthwise=depthwise)

        self.out_channels = ([in_c[0]] if self.use_p2 else []) + [in_c[1], in_c[2], in_c[3]]

    def forward(self, x):
        c2, c3, c4, c5 = self.backbone(x)

        fpn_out0 = self.lateral_conv0(c5)
        f_out0 = torch.cat([F.interpolate(fpn_out0, scale_factor=2.0, mode="nearest"), c4], dim=1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = torch.cat([F.interpolate(fpn_out1, scale_factor=2.0, mode="nearest"), c3], dim=1)
        pan_p3 = self.C3_p3(f_out1)

        outs = []
        if self.use_p2:
            fpn_out2 = self.reduce_conv2(pan_p3)
            f_out2 = torch.cat([F.interpolate(fpn_out2, scale_factor=2.0, mode="nearest"), c2], dim=1)
            pan_p2 = self.C3_p2(f_out2)
            outs.append(pan_p2)
            p_out2 = self.bu_conv_p2(pan_p2)
            pan_p3 = self.C3_n2(torch.cat([p_out2, pan_p3], dim=1))
            outs.append(pan_p3)
        else:
            outs.append(pan_p3)

        p_out1 = self.bu_conv2(pan_p3)
        pan_p4 = self.C3_n3(torch.cat([p_out1, fpn_out1], dim=1))
        outs.append(pan_p4)

        p_out0 = self.bu_conv1(pan_p4)
        pan_p5 = self.C3_n4(torch.cat([p_out0, fpn_out0], dim=1))
        outs.append(pan_p5)
        return outs


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, in_channels, strides, width=1.0, depthwise=False):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        Conv = DWConv if depthwise else BaseConv
        hid_c = int(256 * width)

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for ic in in_channels:
            self.stems.append(BaseConv(ic, hid_c, 1))
            self.cls_convs.append(nn.Sequential(Conv(hid_c, hid_c, 3), Conv(hid_c, hid_c, 3)))
            self.reg_convs.append(nn.Sequential(Conv(hid_c, hid_c, 3), Conv(hid_c, hid_c, 3)))
            self.cls_preds.append(nn.Conv2d(hid_c, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(hid_c, 4, 1))
            self.obj_preds.append(nn.Conv2d(hid_c, 1, 1))

        self._init_biases(prior_prob=1e-2)

    def _init_biases(self, prior_prob):
        b = -math.log((1.0 - prior_prob) / prior_prob)
        for m in self.cls_preds:
            nn.init.constant_(m.bias, b)
        for m in self.obj_preds:
            nn.init.constant_(m.bias, b)

    def forward(self, feats):
        raw_preds = []
        grids = []
        expanded_strides = []

        for lvl, feat in enumerate(feats):
            x = self.stems[lvl](feat)
            cls_feat = self.cls_convs[lvl](x)
            reg_feat = self.reg_convs[lvl](x)

            cls_out = self.cls_preds[lvl](cls_feat)
            reg_out = self.reg_preds[lvl](reg_feat)
            obj_out = self.obj_preds[lvl](reg_feat)

            B, _, H, W = reg_out.shape
            stride = self.strides[lvl]
            out = torch.cat([reg_out, obj_out, cls_out], dim=1)  # [B, 5+nc, H, W]
            out = out.flatten(2).permute(0, 2, 1)  # [B, H*W, 5+nc]
            raw_preds.append(out)

            yv, xv = torch.meshgrid(
                torch.arange(H, device=feat.device, dtype=feat.dtype),
                torch.arange(W, device=feat.device, dtype=feat.dtype),
                indexing="ij",
            )
            grid = torch.stack((xv, yv), 2).reshape(1, H * W, 2)
            grids.append(grid)
            expanded_strides.append(torch.full((1, H * W, 1), float(stride), device=feat.device, dtype=feat.dtype))

        raw_cat = torch.cat(raw_preds, dim=1)
        grid_cat = torch.cat(grids, dim=1)
        stride_cat = torch.cat(expanded_strides, dim=1)
        return raw_cat, grid_cat, stride_cat


class AksaraYOLOX(nn.Module):
    """Wrapper penuh untuk training (loss SimOTA) & ekspor ONNX berformat [B, 4+nc, N]."""

    VARIANTS = {
        "yolox-nano": dict(dep_mul=0.33, wid_mul=0.25, depthwise=True),
        "yolox-tiny": dict(dep_mul=0.33, wid_mul=0.375, depthwise=False),
        "yolox-s": dict(dep_mul=0.33, wid_mul=0.50, depthwise=False),
    }

    def __init__(self, num_classes=55, variant="yolox-tiny", use_p2=True, export_onnx=False):
        super().__init__()
        cfg = self.VARIANTS[variant]
        self.num_classes = num_classes
        self.variant = variant
        self.use_p2 = use_p2
        self.export_onnx = export_onnx
        strides = ([4] if use_p2 else []) + [8, 16, 32]
        self.neck = YOLOXPAFPN(**cfg, use_p2=use_p2)
        self.head = YOLOXHead(num_classes, self.neck.out_channels, strides, width=cfg["wid_mul"], depthwise=cfg["depthwise"])

    def forward(self, x):
        feats = self.neck(x)
        raw, grids, strides = self.head(feats)
        if not self.export_onnx and self.training:
            return raw, grids, strides

        # Decode menjadi [B, 4+nc, N] agar langsung kompatibel dengan aksara_yolo.onnx / app.js / eval
        xy = (raw[..., :2] + grids) * strides
        wh = torch.exp(torch.clamp(raw[..., 2:4], max=12.0)) * strides
        obj = torch.sigmoid(raw[..., 4:5])
        cls_p = torch.sigmoid(raw[..., 5:])
        scores = obj * cls_p  # [B, N, nc]
        decoded = torch.cat([xy, wh, scores], dim=-1)  # [B, N, 4+nc]
        return decoded.permute(0, 2, 1).contiguous()


def bboxes_iou_matrix(bboxes_a, bboxes_b):
    """Matriks IoU untuk kotak [cx, cy, w, h]."""
    tl_a = bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] * 0.5
    br_a = bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] * 0.5
    tl_b = bboxes_b[None, :, :2] - bboxes_b[None, :, 2:] * 0.5
    br_b = bboxes_b[None, :, :2] + bboxes_b[None, :, 2:] * 0.5

    tl = torch.maximum(tl_a, tl_b)
    br = torch.minimum(br_a, br_b)
    en = (tl < br).all(dim=2).float()
    area_i = torch.prod(torch.clamp(br - tl, min=0.0), dim=2) * en
    area_a = torch.prod(torch.clamp(bboxes_a[:, 2:], min=1e-6), dim=1)[:, None]
    area_b = torch.prod(torch.clamp(bboxes_b[:, 2:], min=1e-6), dim=1)[None, :]
    return area_i / torch.clamp(area_a + area_b - area_i, min=1e-6)


def ciou_loss(pred_boxes, target_boxes):
    """CIoU loss untuk kotak [cx, cy, w, h]."""
    px1, py1 = pred_boxes[:, 0] - pred_boxes[:, 2] * 0.5, pred_boxes[:, 1] - pred_boxes[:, 3] * 0.5
    px2, py2 = pred_boxes[:, 0] + pred_boxes[:, 2] * 0.5, pred_boxes[:, 1] + pred_boxes[:, 3] * 0.5
    tx1, ty1 = target_boxes[:, 0] - target_boxes[:, 2] * 0.5, target_boxes[:, 1] - target_boxes[:, 3] * 0.5
    tx2, ty2 = target_boxes[:, 0] + target_boxes[:, 2] * 0.5, target_boxes[:, 1] + target_boxes[:, 3] * 0.5

    ix1, iy1 = torch.maximum(px1, tx1), torch.maximum(py1, ty1)
    ix2, iy2 = torch.minimum(px2, tx2), torch.minimum(py2, ty2)
    inter = torch.clamp(ix2 - ix1, min=0.0) * torch.clamp(iy2 - iy1, min=0.0)
    area_p = torch.clamp(px2 - px1, min=0.0) * torch.clamp(py2 - py1, min=0.0)
    area_t = torch.clamp(tx2 - tx1, min=0.0) * torch.clamp(ty2 - ty1, min=0.0)
    union = torch.clamp(area_p + area_t - inter, min=1e-6)
    iou = inter / union

    cx1, cy1 = torch.minimum(px1, tx1), torch.minimum(py1, ty1)
    cx2, cy2 = torch.maximum(px2, tx2), torch.maximum(py2, ty2)
    c2 = torch.clamp((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2, min=1e-6)
    rho2 = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    pw = torch.clamp(pred_boxes[:, 2], min=1e-4)
    ph = torch.clamp(pred_boxes[:, 3], min=1e-4)
    tw = torch.clamp(target_boxes[:, 2], min=1e-4)
    th = torch.clamp(target_boxes[:, 3], min=1e-4)
    v = (4.0 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / torch.clamp(1.0 - iou + v, min=1e-6)
    return 1.0 - iou + rho2 / c2 + alpha * v


def compute_yolox_loss(raw, grids, strides, targets_list, num_classes=55, label_smoothing=0.03, class_weights=None):
    """
    Hitung loss YOLOX dengan SimOTA matching (selalu dieksekusi dalam FP32 agar aman untuk CUDA AMP).
    targets_list: list len B dari tensor [M_i, 5] berisi (cid, cx_px, cy_px, w_px, h_px).
    """
    with torch.amp.autocast(device_type=raw.device.type, enabled=False):
        return _compute_yolox_loss_fp32(
            raw.float(), grids.float(), strides.float(), targets_list,
            num_classes=num_classes, label_smoothing=label_smoothing, class_weights=class_weights,
        )


def _compute_yolox_loss_fp32(raw, grids, strides, targets_list, num_classes=55, label_smoothing=0.03, class_weights=None):
    B, N, _ = raw.shape
    device = raw.device
    xy_pred = (raw[..., :2] + grids) * strides
    wh_pred = torch.exp(torch.clamp(raw[..., 2:4], max=10.0)) * strides
    boxes_pred = torch.cat([xy_pred, wh_pred], dim=-1)  # [B, N, 4]
    obj_logits = raw[..., 4:5]
    cls_logits = raw[..., 5:]

    grid_centers = (grids[0] + 0.5) * strides[0]  # [N, 2]
    stride_flat = strides[0, :, 0]  # [N]

    total_reg, total_obj, total_cls = 0.0, 0.0, 0.0
    num_fg = 0

    for b in range(B):
        t = targets_list[b]
        if t.numel() == 0:
            obj_tgt = torch.zeros_like(obj_logits[b])
            total_obj = total_obj + F.binary_cross_entropy_with_logits(obj_logits[b], obj_tgt, reduction="sum")
            continue

        gt_cls = t[:, 0].long()
        gt_boxes = t[:, 1:5]  # [M, 4] cx,cy,w,h
        M = gt_boxes.shape[0]

        # In-boxes & In-center (radius 2.5 * stride) mask
        gt_tl = gt_boxes[:, None, :2] - gt_boxes[:, None, 2:] * 0.5
        gt_br = gt_boxes[:, None, :2] + gt_boxes[:, None, 2:] * 0.5
        in_box = ((grid_centers[None, :, 0] > gt_tl[:, :, 0]) &
                  (grid_centers[None, :, 0] < gt_br[:, :, 0]) &
                  (grid_centers[None, :, 1] > gt_tl[:, :, 1]) &
                  (grid_centers[None, :, 1] < gt_br[:, :, 1]))

        rad = 2.5 * stride_flat[None, :]
        in_ctr = ((torch.abs(grid_centers[None, :, 0] - gt_boxes[:, None, 0]) < rad) &
                  (torch.abs(grid_centers[None, :, 1] - gt_boxes[:, None, 1]) < rad))

        is_in_boxes_and_ctr = in_box & in_ctr
        is_in_any = in_box | in_ctr
        fg_mask = is_in_any.any(dim=0)
        fg_idx = torch.where(fg_mask)[0]
        if fg_idx.numel() == 0:
            obj_tgt = torch.zeros_like(obj_logits[b])
            total_obj = total_obj + F.binary_cross_entropy_with_logits(obj_logits[b], obj_tgt, reduction="sum")
            continue

        p_boxes_fg = boxes_pred[b, fg_idx]  # [K, 4]
        with torch.no_grad():
            pair_iou = bboxes_iou_matrix(gt_boxes, p_boxes_fg)  # [M, K]
            iou_loss_mat = -torch.log(pair_iou + 1e-8)

            gt_one_hot = F.one_hot(gt_cls, num_classes).float()  # [M, nc]
            p_cls_fg = (torch.sigmoid(cls_logits[b, fg_idx]) * torch.sigmoid(obj_logits[b, fg_idx])).sqrt()
            p_cls_fg = torch.nan_to_num(p_cls_fg, nan=0.5, posinf=1.0 - 1e-5, neginf=1e-5)
            p_cls_fg = torch.clamp(p_cls_fg, 1e-5, 1.0 - 1e-5)
            cls_cost = F.binary_cross_entropy(
                p_cls_fg.unsqueeze(0).expand(M, -1, -1),
                gt_one_hot.unsqueeze(1).expand(-1, fg_idx.numel(), -1),
                reduction="none",
            ).sum(-1)

            in_cand = is_in_boxes_and_ctr[:, fg_idx]
            cost = cls_cost + 3.0 * iou_loss_mat + 1e5 * (~in_cand).float()

            # Dynamic-k estimation
            n_cand_k = min(10, fg_idx.numel())
            topk_ious, _ = torch.topk(pair_iou, n_cand_k, dim=1)
            dynamic_ks = torch.clamp(topk_ious.sum(dim=1).int(), min=1)

            matching = torch.zeros_like(cost, dtype=torch.bool)
            for m_idx in range(M):
                _, pos_k = torch.topk(cost[m_idx], k=int(dynamic_ks[m_idx]), largest=False)
                matching[m_idx, pos_k] = True

            # Resolve anchor assigned to >1 GT by min cost
            multi = matching.sum(dim=0) > 1
            if multi.any():
                _, min_gt = torch.min(cost[:, multi], dim=0)
                matching[:, multi] = False
                matching[min_gt, multi] = True

            anchor_has_gt = matching.any(dim=0)
            matched_fg_idx = fg_idx[anchor_has_gt]
            matched_gt_idx = matching[:, anchor_has_gt].float().argmax(dim=0)

        if matched_fg_idx.numel() == 0:
            obj_tgt = torch.zeros_like(obj_logits[b])
            total_obj = total_obj + F.binary_cross_entropy_with_logits(obj_logits[b], obj_tgt, reduction="sum")
            continue

        num_fg += matched_fg_idx.numel()
        matched_gt_boxes = gt_boxes[matched_gt_idx]
        matched_gt_cls = gt_cls[matched_gt_idx]
        matched_ious = bboxes_iou_matrix(matched_gt_boxes, boxes_pred[b, matched_fg_idx]).diag().detach()

        reg_l = ciou_loss(boxes_pred[b, matched_fg_idx], matched_gt_boxes).sum()
        total_reg = total_reg + 5.0 * reg_l

        obj_tgt = torch.zeros_like(obj_logits[b])
        obj_tgt[matched_fg_idx, 0] = 1.0
        total_obj = total_obj + F.binary_cross_entropy_with_logits(obj_logits[b], obj_tgt, reduction="sum")

        cls_tgt = F.one_hot(matched_gt_cls, num_classes).float() * (1.0 - label_smoothing) + (label_smoothing / num_classes)
        cls_tgt = cls_tgt * matched_ious.unsqueeze(1)
        bce_cls = F.binary_cross_entropy_with_logits(cls_logits[b, matched_fg_idx], cls_tgt, reduction="none")
        if class_weights is not None:
            w_m = class_weights[matched_gt_cls].unsqueeze(1)
            bce_cls = bce_cls * w_m
        total_cls = total_cls + bce_cls.sum()

    if num_fg == 0:
        neg_denom = max(16.0, float(B) * 16.0)
        obj_loss = total_obj / neg_denom if torch.is_tensor(total_obj) else raw.sum() * 0.0
        return obj_loss, {
            "reg": 0.0,
            "obj": float(obj_loss.detach()),
            "cls": 0.0,
            "fg": 0,
        }

    denom = max(1, num_fg)
    return (total_reg + total_obj + total_cls) / denom, {
        "reg": float(total_reg.detach() / denom) if torch.is_tensor(total_reg) else 0.0,
        "obj": float(total_obj.detach() / denom) if torch.is_tensor(total_obj) else 0.0,
        "cls": float(total_cls.detach() / denom) if torch.is_tensor(total_cls) else 0.0,
        "fg": num_fg,
    }
