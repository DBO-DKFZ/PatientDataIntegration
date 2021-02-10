from argparse import Namespace 
from jsonargparse import ArgumentParser, ActionConfigFile
from ExperimentSetup_v2 import Experiment # look in folder "PDI_classes_and_functions" for these scripts 
import configs # configs include the path to the images, patient data and trained models; the config-file is not included in this repository 

def main(hparams):
    
    experiment = Experiment(hparams)
    
if __name__ == '__main__': 
    parser = ArgumentParser(add_help = False)
    parser.add_argument('--cfg', action=ActionConfigFile)
    
    parser.add_argument('--experiment_id', type=str, default='test')
    parser.add_argument('--setting', type=str, default='K')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--rs', type=int, default=1) # random seed 
    parser.add_argument('--nr_tiles', type=int, default = 100)
    parser.add_argument('--dataset', type=str, nargs='+', default= ['lab2', 'lab1', 'lab3']) 
    parser.add_argument('--splits', type=int, default=5)
    parser.add_argument('--test_flag', type=bool, default=False)

    parser.add_argument('--mode', type=str, default='base') # 'approach_1', 'approach_2', 'approach_3'
    parser.add_argument('--encoding_scheme', type=str, default='unscaled') # 'scale01', 'onehot'
    parser.add_argument('--metadata_list', type=str, nargs='+', default= ['age', 'gender', 'location']) 

    
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=5)
    
    parser.add_argument('--meta_out', type=int, default=512)
    parser.add_argument('--hidden1', type=int, default=2048) # 512? 
    parser.add_argument('--dropout', type=float, default=0.3)
    
  
    parser.add_argument('--epoch_fine_tuning', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--runs', type=int, default=5)
    
    parser.add_argument('--lr_unfrozen', type=int, default=0.0001)
    parser.add_argument('--lr_frozen', type=int, default=0.001) # 0.001
    
    parser.add_argument('--lr_meta_unfrozen', type= int, default=0.0001)
    parser.add_argument('--lr_fc_unfrozen', type= int, default=0.001)
    parser.add_argument('--lr_fine', type= int, default=0.00001)
    parser.add_argument('--end_epoch_p1', type= int, default=5)
    parser.add_argument('--end_epoch_p2', type= int, default=10)
    parser.add_argument('--matched', type=bool, default=False)
   
    rp = configs.get_result_path()
    parser.add_argument('--overview_results', type=str, default= rp + 'overview_results_finals.csv')
    

    hparams = parser.parse_args()
    hparams = Namespace(**hparams.__dict__)
    main(hparams)

