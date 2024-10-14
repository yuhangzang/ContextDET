import torch
import torch.distributed as dist


class AccEvaluator(object):
    def __init__(self, k_list=[1, ]):
        self.k_list = k_list
        self.correct = torch.zeros(len(k_list), 1).float().cuda()
        self.count = torch.zeros(len(k_list), 1).float().cuda()

    def update(self, outputs, mask_infos, tokenizer):
        pred_tokens_list = []
        for b in range(len(mask_infos)):
            pred_tokens_b = []
            for (start, end), name in mask_infos[b].items():
                mask_token_logits = outputs[b, start:end + 1, :]
                for ind, k in enumerate(self.k_list):
                    top_5_tokens = torch.topk(mask_token_logits, k, dim=1).indices.T.tolist()
                    pred_tokens = [tokenizer.decode(token).strip() for token in top_5_tokens]
                    if k == self.k_list[-1]:
                        pred_tokens_b.append(pred_tokens)
                    if name in pred_tokens:
                        self.correct[ind] += 1
                    self.count[ind] += 1
            pred_tokens_list.append(pred_tokens_b)
        return pred_tokens_list

    def synchronize_between_processes(self):
        dist.all_reduce(self.correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.count, op=dist.ReduceOp.SUM)

    def summarize(self):
        for ind, k in enumerate(self.k_list):
            correct = self.correct[ind]
            count = self.count[ind]
            acc = float(correct / count)
            print(f'Acc@Top{k}: {float(correct)} / {float(count)} = {acc}')
