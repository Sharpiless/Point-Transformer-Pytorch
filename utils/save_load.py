import torch
import collections

def load_pretrained(model, weights):

    state_dict = torch.load(weights)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    src_state_dict = model.state_dict()
    fsd = collections.OrderedDict()
    for key, value in state_dict.items():
        if key in state_dict:
            if value.shape == src_state_dict[key].shape:
                fsd[key] = value
                print('-[INFO] sucessfully loaded', key)
            else:
                print('-[WARN] shape mis-match', key)
        else:
            print('-[WARN] unexcepted', key)
    model.load_state_dict(fsd, strict=False)
