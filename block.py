import fastai
from fastai import *


class linear_block(nn.Module):
    def __init__(self, ip_sz, op_sz, drop = None, bias = True, initrange = None,
                eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.op_sz = op_sz
        self.ln = nn.Linear(ip_sz, op_sz, bias = bias)
        kaiming_normal_(self.ln.weight.data)
        if initrange is not None: self.ln.weight.data.uniform_(-initrange, initrange)
        if bias: self.ln.bias.data.zero_()
        if drop is not None:
            self.bn = nn.BatchNorm1d(op_sz, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
            self.drp = nn.Dropout(drop)
        else: self.drp = None
        
    def forward(self, x): return self.ln(x.float()) if self.drp is None else self.drp(self.bn(F.relu(self.ln(x.float()))))


class attention_block(nn.Module):
    def __init__(self, input_sz = 64, hidden_sz = 64, tp = 'fea'):
        super().__init__()
        self.tp = tp
        self.ln_h = linear_block(hidden_sz, input_sz)
        self.ln_i = linear_block(input_sz, input_sz)
        self.ln_w = linear_block(input_sz, input_sz)
        self.v = self.rand_p(input_sz, input_sz)
    
    @staticmethod
    def rand_p(*sz): return nn.Parameter(torch.randn(sz)/math.sqrt(sz[0]))
              
    def forward(self, inpt, hidden):
        wh = self.ln_h(hidden.permute(1,0,2))
        wi = self.ln_i(inpt)
        u = torch.tanh(wh + wi)
        if self.tp == 'fea':
            att = F.softmax((u@self.v).sum(1), 1)
            w = (att.double().unsqueeze(1)*inpt.double()).sum(1)
        elif self.tp == 'seq':
            att = F.softmax(u@self.v, 1)
            w = (att.double()*inpt.double()).sum(1)
        w = self.ln_w(w.float())
        return w, att


class rnn_block(nn.Module):
    def __init__(self, input_sz,hidden_sz, n_rnn, hidden_drp=0.2, weight_drp=0.5, output_drp = 0.2):
        super().__init__()
        self.n_rnn = n_rnn
        self.hidden_drps = nn.ModuleList([RNNDropout(hidden_drp) for l in range(n_rnn)])
        
        self.rnns = [nn.LSTM(input_sz if l == 0 else hidden_sz, hidden_sz, 1, batch_first = True) for l in range(n_rnn)]
        self.rnns = [WeightDropout(rnn, weight_drp) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        
        self.output_drps = RNNDropout(output_drp)
    
    def forward(self, raw_output):
        hiddens_states,cells_states,raw_outputs,outputs = [],[],[],[]
        for l, (rnn,hidden_drp) in enumerate(zip(self.rnns, self.hidden_drps)):
            raw_output, (hidden_state, cell_state) = rnn(raw_output.float())
            hiddens_states.append(hidden_state); cells_states.append(cell_state); raw_outputs.append(raw_output)
            if l != self.n_rnn - 1: raw_output = hidden_drp(raw_output)
            outputs.append(raw_output)
        return outputs, raw_outputs, hiddens_states, cells_states


