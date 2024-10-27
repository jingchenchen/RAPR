import argparse

parser = argparse.ArgumentParser()


YML_PATH = {
    "mit-states": './config/mit-states.yml',
    "ut-zappos": './config/ut-zappos.yml',
    "cgqa": './config/cgqa.yml'
}

#model config
parser.add_argument("--lr", help="learning rate", type=float, default=5e-05)
parser.add_argument("--dataset", help="name of the dataset", type=str, default='mit-states')
parser.add_argument("--model_name",  default= 'rapr', type=str)
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-05)
parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-L/14")
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
parser.add_argument("--epoch_start", help="start epoch", default=0, type=int)
parser.add_argument("--train_batch_size", help="train batch size", default=48, type=int)
parser.add_argument("--eval_batch_size", help="eval batch size", default=16, type=int)
parser.add_argument("--context_length", help="sets the context length of the clip model", default=8, type=int)
parser.add_argument("--attr_dropout", help="add dropout to attributes", type=float, default=0.3)
parser.add_argument("--save_path", help="save path", type=str)
parser.add_argument("--save_every_n", default=5, type=int, help="saves the model every n epochs")
parser.add_argument("--save_model", help="indicate if you want to save the model state dict()", action="store_true",default=True)
parser.add_argument("--test_epoch", type=str, default='best')
parser.add_argument("--load_model", default=None, help="load the trained model")
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument("--gradient_accumulation_steps", help="number of gradient accumulation steps", default=1, type=int)
parser.add_argument("--open_world", help="evaluate on open world setup", default= False)
parser.add_argument("--bias", help="eval bias", type=float, default=1e3)
parser.add_argument("--topk", help="eval topk", type=int, default=1)
parser.add_argument('--threshold', type=float, help="optional threshold")
parser.add_argument('--threshold_trials', type=int, default=50, help="how many threshold values to try")
parser.add_argument("--att_obj_w", type=float, default=0.0)
parser.add_argument("--exp_id", type=str, default='1')
parser.add_argument("--LR_K", type=int, default=1)
parser.add_argument("--projection", type=str, default='mlp')
parser.add_argument("--res_w_vis", type=float, default=0.8)
parser.add_argument("--res_w_vis_obj", type=float, default=0.8)
parser.add_argument("--res_w_vis_att", type=float, default=0.8)
parser.add_argument("--database_path", type=str, default='data/database/')
parser.add_argument("--retrieval_topk", type=int, default=16)
parser.add_argument("--retrieval_temperature", type=float, default=1.0)
parser.add_argument("--retrieval_weight", type=float, default=0.1)
parser.add_argument("--build_database_every", type=int, default=1)
parser.add_argument("--retrieval_loss_topk", type=int, default=1)
parser.add_argument("--retrieval_loss_weight", type=float, default=1.0)
parser.add_argument("--debias_loss_weight",  type=float, default=1.0)