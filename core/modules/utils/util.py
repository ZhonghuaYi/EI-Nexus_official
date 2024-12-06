import torch
import torch.nn.functional as F


class Padder:
    def __init__(self, shape, p):
        self.shape = shape
        self.p = p
        h, w = shape[-2:]
        h_padding = (((h // p) + 1) * p - h) % p
        w_padding = (((w // p) + 1) * p - w) % p
        # pad the input tensor to the nearest multiple of p
        h_padding0, h_padding1 = h_padding // 2, h_padding - h_padding // 2
        w_padding0, w_padding1 = w_padding // 2, w_padding - w_padding // 2
        self.padding_size = (w_padding0, w_padding1, h_padding0, h_padding1)
    
    def pad(self, *args):
        """
        Pad the input tensors to the same size.
        Args:
            *args (torch.Tensor): tensors to pad.
        Returns:
            padded_args (list): list of padded tensors.
        """
        out = []
        for arg in args:
            if arg.dtype == torch.bool:
                out.append(F.pad(arg, self.padding_size, mode='constant'))
            else:
                out.append(F.pad(arg, self.padding_size, mode='replicate'))
        
        return out
    
    def unpad(self, *args):
        """
        Unpad the input tensors to the same size.
        Args:
            *args (torch.Tensor): tensors to unpad.
        Returns:
            unpadded_args (list): list of unpadded tensors.
        """
        
        out = []
        
        for arg in args:
            h, w = arg.shape[-2:]
            c = [self.padding_size[2], h - self.padding_size[3], self.padding_size[0], w - self.padding_size[1]]
            out.append(arg[..., c[0]:c[1], c[2]:c[3]].clone().contiguous())
        
        return out
    
    def unpad_positions(self, positions_list, ordering='xy'):
        assert ordering in ('xy', 'yx')
        out = []
        
        for positions in positions_list:
            new_positions = positions.clone()
            if ordering == 'xy':
                new_positions[..., 0] = positions[..., 0] - self.padding_size[0]
                new_positions[..., 1] = positions[..., 1] - self.padding_size[2]
            elif ordering == 'yx':
                new_positions[..., 0] = positions[..., 0] - self.padding_size[2]
                new_positions[..., 1] = positions[..., 1] - self.padding_size[0]
            out.append(new_positions)
        
        return out

