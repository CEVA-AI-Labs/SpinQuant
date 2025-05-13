import torch
from collections import OrderedDict


def export(state_dict, group_size):
    """ Exports state_dict to LiteML format."""
    liteml_state_dict = dict()
    for key, value in state_dict['w_quantizers'].items():
        prefix = key.split('.module')[0]
        weights_quantizer_prefix = f"_model._model.{prefix}._weights_quantizer.obs"
        if group_size > -1:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale[None, :, :, None]
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero[None, :, :, None]
        else:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero


    for key, value in state_dict['model'].items():
        last, before_last = key.split('.')[-1], key.split('.')[-2]
        if 'layernorm' in key:
            key_liteml = f"_model._model.{key}"
            liteml_state_dict[key_liteml] = value
        elif key == 'model.embed_tokens.weight' or key == 'model.norm.weight':
            key_liteml = f"_model._model.{key}"
            liteml_state_dict[key_liteml] = value
        elif last == 'weight' and before_last != 'module':
            key_liteml = f"_model._model.{key.replace('weight','_model.weight')}"
            liteml_state_dict[key_liteml] = value

    liteml_state_dict = OrderedDict(sorted(liteml_state_dict.items()))
    return liteml_state_dict


def export2(state_dict, group_size):
    """
    Exports state_dict to LiteML format where the state_dict's path is provided in the config.yaml file, thus
    we remove the ._model._model prefix compared to the export function above.
    """
    liteml_state_dict = dict()
    for key, value in state_dict['w_quantizers'].items():
        prefix = key.split('.module')[0]
        weights_quantizer_prefix = f"{prefix}._weights_quantizer.obs"
        if group_size > -1:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale[None, :, :, None]
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero[None, :, :, None]
        else:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero


    for key, value in state_dict['model'].items():
        last, before_last = key.split('.')[-1], key.split('.')[-2]
        if 'layernorm' in key:
            liteml_state_dict[key] = value
        elif key == 'model.embed_tokens.weight' or key == 'model.norm.weight':
            liteml_state_dict[key] = value
        elif last == 'weight' and before_last != 'module':
            key_liteml = f"{key.replace('weight','_model.weight')}"
            liteml_state_dict[key_liteml] = value

    liteml_state_dict = OrderedDict(sorted(liteml_state_dict.items()))
    return liteml_state_dict



def export_with_TrueQuantRMSNorm(state_dict, group_size):
    """
    Exports state_dict to LiteML format for a model that uses TrueQuantRMSNorm layer.
    """
    liteml_state_dict = dict()
    for key, value in state_dict['w_quantizers'].items():
        prefix = key.split('.module')[0]
        weights_quantizer_prefix = f"{prefix}._weights_quantizer.obs"
        if group_size > -1:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale[None, :, :, None]
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero[None, :, :, None]
        else:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero


    for key, value in state_dict['model'].items():
        last, before_last = key.split('.')[-1], key.split('.')[-2]
        if 'layernorm' in key or key == 'model.norm.weight':
            key = key.replace('weight', '_RMSnorm.weight')  # change RMSnorm key according to TrueQuantRMSNorm implementation
            liteml_state_dict[key] = value
        elif key == 'model.embed_tokens.weight':
            liteml_state_dict[key] = value
        elif last == 'weight' and before_last != 'module':
            key_liteml = f"{key.replace('weight','_model.weight')}"
            liteml_state_dict[key_liteml] = value

    liteml_state_dict = OrderedDict(sorted(liteml_state_dict.items()))
    return liteml_state_dict

spinquant_path = 'saved_models/spinquant_gptq_group128_chat.pth'
group_size = 128
state_dict = torch.load(spinquant_path)
out_state_dict = export_with_TrueQuantRMSNorm(state_dict, group_size)
torch.save(out_state_dict, 'saved_models/liteml_spinquant_gptq_group128_TrueQuantRMSNorm_chat.pth')
print('Done')