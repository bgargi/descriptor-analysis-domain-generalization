def extract_pacs_data(data, merge_groups=True, transform=None, ):
    zs, ys, preds, gs, logits = data['feature'], data['label'], data['pred'], data['group'], data['logits']
    if transform is not None:
        zs = transform(zs)
    #     gs = gs % 2
    return zs, ys, gs, preds, logits