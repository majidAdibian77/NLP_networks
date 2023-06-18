import os
import pickle
from simplet5 import SimpleT5
import pandas as pd

if __name__ == "__main__":
    print("load train and val data..")
    train_data = pd.read_csv(os.path.join('T5', 'data', 'train_data.csv'))
    val_data = pd.read_csv(os.path.join('T5', 'data', 'val_data.csv'))
    
    model = SimpleT5()
    model.from_pretrained(model_type="t5", model_name="t5-base")

    source_max_token_len = len(max(train_data['source_text'], key=lambda x:len(x.split())).split())+20
    target_max_token_len = len(max(train_data['target_text'], key=lambda x:len(x.split())).split())+20
    model.train(train_df=train_data,
                eval_df=val_data, 
                source_max_token_len=source_max_token_len, 
                target_max_token_len=target_max_token_len, 
                outputdir = os.path.join("T5", "model"),
                save_only_last_epoch = True,
                batch_size=32, max_epochs=5, use_gpu=True)