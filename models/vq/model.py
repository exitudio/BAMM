import random

import torch.nn as nn
from models.vq.encdec import Encoder, Decoder
from models.vq.residual_vq import ResidualVQ
import torch
from utils.humanml_utils import HML_UPPER_BODY_MASK, HML_LOWER_BODY_MASK, UPPER_JOINT_Y_MASK
    
class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.is_upperlower = False
        if self.is_upperlower:
            rvqvae_config['code_dim'] = int(code_dim/2)
            if args.dataset_name == 'kit':
                self.nb_joints = 21
                output_dim = 251
                upper_dim = 120        
                lower_dim = 131  
            else:
                self.nb_joints = 22
                output_dim = 263
                upper_dim = 156        
                lower_dim = 107 
            self.encoder_upper = Encoder(upper_dim, int(output_emb_width/2), down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
            self.encoder_lower = Encoder(lower_dim, int(output_emb_width/2), down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
            self.quantizer_upper = ResidualVQ(**rvqvae_config)
            self.quantizer_lower = ResidualVQ(**rvqvae_config)
            self.register_buffer('mean_upper', torch.tensor([0.1216, 0.2488, 0.2967, 0.5027, 0.4053, 0.4100, 0.5703, 0.4030, 0.4078, 0.1994, 0.1992, 0.0661, 0.0639], dtype=torch.float32))
            self.register_buffer('std_upper', torch.tensor([0.0164, 0.0412, 0.0523, 0.0864, 0.0695, 0.0703, 0.1108, 0.0853, 0.0847, 0.1289, 0.1291, 0.2463, 0.2484], dtype=torch.float32))
        else:
            self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
            self.quantizer = ResidualVQ(**rvqvae_config)
        
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x):
        if self.is_upperlower:
            x = x.float()
            x = self.shift_upper_down(x)

            upper_emb = x[..., HML_UPPER_BODY_MASK]
            lower_emb = x[..., HML_LOWER_BODY_MASK]
            upper_emb = self.preprocess(upper_emb)
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb, upper_code_idx, upper_commit_loss, upper_perplexity = self.quantizer_upper(upper_emb, sample_codebook_temp=0.5)

            lower_emb = self.preprocess(lower_emb)
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb, lower_code_idx, lower_commit_loss, lower_perplexity = self.quantizer_lower(lower_emb, sample_codebook_temp=0.5)
            commit_loss = upper_commit_loss + lower_commit_loss
            perplexity = upper_perplexity + lower_perplexity

            x_quantized = torch.cat([upper_emb, lower_emb], dim=1)
            x_out = self.decoder(x_quantized)
            return x_out, commit_loss, perplexity
        else:
            x_in = self.preprocess(x)
            # Encode
            x_encoder = self.encoder(x_in)

            ## quantization
            # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
            #                                                                 force_dropout_index=0) #TODO hardcode
            x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)

            # print(code_idx[0, :, 1])
            ## decoder
            x_out = self.decoder(x_quantized)
            # x_out = self.postprocess(x_decoder)
            return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out
    
    def normalize(self, data):
        return (data - self.moment['mean']) / self.moment['std']
    
    def denormalize(self, data):
        return data * self.moment['std'] + self.moment['mean']
    
    def normalize_upper(self, data):
        return (data - self.mean_upper) / self.std_upper
    
    def denormalize_upper(self, data):
        return data * self.std_upper + self.mean_upper
    
    def shift_upper_down(self, data):
        data = data.clone()
        data = self.denormalize(data)
        shift_y = data[..., 3:4].clone()
        data[..., UPPER_JOINT_Y_MASK] -= shift_y
        _data = data.clone()
        data = self.normalize(data)
        data[..., UPPER_JOINT_Y_MASK] = self.normalize_upper(_data[..., UPPER_JOINT_Y_MASK])
        return data
    
    def shift_upper_up(self, data):
        _data = data.clone()
        data = self.denormalize(data)
        data[..., UPPER_JOINT_Y_MASK] = self.denormalize_upper(_data[..., UPPER_JOINT_Y_MASK])
        shift_y = data[..., 3:4].clone()
        data[..., UPPER_JOINT_Y_MASK] += shift_y
        data = self.normalize(data)
        return data

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)