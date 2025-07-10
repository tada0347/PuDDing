from utils.onoff_utils import onoff_opt, onoff_llama, onoff_phi

def block_replace(model):
    if 'opt' in model.name.lower():
        model = onoff_opt.block_replace(model)
    elif 'llama' in model.name.lower() or 'vicuna' in model.name.lower():
        model = onoff_llama.block_replace(model)
    elif 'phi' in model.name.lower():
        model = onoff_phi.block_replace(model)
    else:
        print(sdfs)
    return model

def turn_off(model, block_idx):
    # print(model.name)

    if 'opt' in model.name.lower():
        onoff_opt.turn_off(model, block_idx)
    elif 'llama' in model.name.lower() or 'vicuna' in model.name.lower():
        onoff_llama.turn_off(model, block_idx)
    elif 'phi' in model.name.lower():
        onoff_phi.turn_off(model, block_idx)

def turn_on(model, block_idx):
    if 'opt' in model.name.lower():
        onoff_opt.turn_on(model, block_idx)
    elif 'llama' in model.name.lower() or 'vicuna' in model.name.lower():
        onoff_llama.turn_on(model, block_idx)
    elif 'phi' in model.name.lower():
        onoff_phi.turn_on(model, block_idx)


def scan(model, num_blocks):

    if 'opt' in model.name.lower():
        onoff_opt.scan(model, num_blocks)
    elif 'llama' in model.name.lower() or 'vicuna' in model.name.lower():
        onoff_llama.scan(model, num_blocks)
    elif 'phi' in model.name.lower():
        onoff_phi.scan(model, numblocks)
    
